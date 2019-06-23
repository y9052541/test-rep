import tvm
import sys
import os
import signal
import logging
import threading
from tvm import autotvm
from multiprocessing import Process

global function

@autotvm.template
def gemm_tuning(batch, N, L, M):
    bn = 32
    A = tvm.placeholder((batch, N, L), name='A', dtype='float32')
    B = tvm.placeholder((batch, L, M), name='B', dtype='float32')
    packedB = tvm.compute((batch, N / bn, L, bn), lambda b, x, y, z: B[b, y, x * bn + z], name='packedB')

    k = tvm.reduce_axis((0, L), name='k')
    C = tvm.compute((batch, M, N),
                lambda b, x, y: tvm.sum(A[b, x, k] * packedB[b, y / bn, k, y % bn], axis=k),
                name = 'C')
    s = tvm.create_schedule(C.op)
    
    ##### define space and schedule
    cfg = autotvm.get_config()

    bn = 32
    CC = s.cache_write(C, 'global')

    factor_range = [2, 4, 8, 16, 32, 64]
    cfg.define_knob('tile_factor_x', factor_range)
    cfg.define_knob('tile_factor_y', factor_range)
    bx = cfg['tile_factor_x'].val
    by = cfg['tile_factor_y'].val
    xo, yo, xi, yi = s[C].tile(C.op.axis[1], C.op.axis[2], bx, by)

    s[CC].compute_at(s[C], yo)
    b, xc, yc = s[CC].op.axis
    k, = s[CC].op.reduce_axis
    """cfg.define_split("split_k", k, num_outputs=2)
    ko, ki = cfg["split_k"].apply(s, CC, k)"""

    k_num_outputs_range = [2, 3, 4, 5, 6, 7, 8]
    cfg.define_knob('k_outputs', k_num_outputs_range)
    k_outputs = cfg['k_outputs'].val
    cfg.define_split("split_k", k, policy='all', num_outputs=k_outputs)
    k_list = cfg["split_k"].apply(s, CC, k)
    cfg.define_reorder("reorder_k", axes=[xc, yc] + k_list, policy='all')
    cfg["reorder_k"].apply(s, CC, [xc, yc] + k_list)

    """cfg.define_reorder("reorder_k", [ko, xc, ki, yc], policy='all')
    cfg["reorder_k"].apply(s, CC, s[CC].op.axis)"""
    # s[CC].reorder(ko, xc, ki, yc)

    k_unroll_id = list(range(k_outputs))
    # print(len(k_list))
    cfg.define_knob('k_unroll', k_unroll_id)
    k_id = cfg['k_unroll'].val
    # print(type(k_id))
    s[CC].unroll(k_list[k_id])
    # s[CC].unroll(ki)

    cfg.define_knob('vector_dim', [0, 1])
    vector_id = cfg['vector_dim'].val
    if vector_id == 0:
        s[CC].vectorize(yc)
    else:
        s[CC].vectorize(xc)
    # s[CC].vectorize(yc)

    parallel_list = [xo, yo, xi, yi]
    cfg.define_knob('parallel_C', list(range(len(parallel_list))))
    parallel_C_id = cfg['parallel_C'].val
    # print(len(parallel_list))
    s[C].parallel(parallel_list[parallel_C_id])
    # s[C].parallel(xo)
    return s, [A, B, C]

@autotvm.template
def conv2d_turning(*args):
    global function
    ops, bufs = function(*args)
    data, kernel, conv = bufs
    s = tvm.create_schedule(conv.op)
        

    ##### space definition begin #####
    if rx==1 and ry==1:
        C=tvm.compute((n,f,v,x),
            lambda i,j,k,l:tvm.sum(data[i][j+rc][k][l]*kernel[j][rc][0][0],axis=rc),name='C')
        ss= tvm.create_schedule(C.op)
        n, f, y, x = ss[C].op.axis
        rc= ss[C].op.reduce_axis
        cfg = autotvm.get_config()
        cfg.define_split("tile_f", f, num_outputs=4)
        cfg.define_split("tile_y", y, num_outputs=4)
        cfg.define_split("tile_x", x, num_outputs=4)
        cfg.define_split("tile_rc", rc, num_outputs=3)
        cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
        ##### space definition end #####

        # inline padding
        pad_data = ss[C].op.input_tensors[0]
        ss[pad_data].compute_inline()
        data, raw_data = pad_data, data

        output = C
        OL = ss.cache_write(C, 'global')
        # tile and bind spatial axes
        n, f, y, x = ss[output].op.axis
        bf, vf, tf, fi = cfg["tile_f"].apply(ss, output, f)
        by, vy, ty, yi = cfg["tile_y"].apply(ss, output, y) 
        bx, vx, tx, xi = cfg["tile_x"].apply(ss, output, x)
        kernel_scope = n  # this is the scope to attach global config inside this kernel

        ss[output].reorder(n, bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
        fuse = ss[output].fuse(yi, xi)
        ss[output].vectorize(fuse)
        ss[OL].compute_at(ss[output], tx)

        # tile reduction axes
        n, f, y, x = ss[OL].op.axis
        rc,= ss[OL].op.reduce_axis
        rco, rcm, rci = cfg['tile_rc'].apply(ss, OL, rc)
        ss[OL].reorder(n, rco,rcm, rci, f, y, x)

        n, f, y, x = ss[C].op.axis
        ss[C].parallel(n)

        #print(tvm.lower(s, [data, kernel, conv], simple_mode=True))

        return ss, [raw_data, kernel, C]



    else:
        n, f, y, x = s[conv].op.axis
        rc, ry, rx = s[conv].op.reduce_axis
       

        cfg = autotvm.get_config()
        cfg.define_split("tile_f", f, num_outputs=4)
        cfg.define_split("tile_y", y, num_outputs=4)
        cfg.define_split("tile_x", x, num_outputs=4)
        cfg.define_split("tile_rc", rc, num_outputs=3)
        cfg.define_split("tile_ry", ry, num_outputs=3)
        cfg.define_split("tile_rx", rx, num_outputs=3)
        cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
        ##### space definition end #####

        # inline padding
        pad_data = s[conv].op.input_tensors[0]
        s[pad_data].compute_inline()
        data, raw_data = pad_data, data
    


        output = conv
        OL = s.cache_write(conv, 'global')
        # tile and bind spatial axes
        n, f, y, x = s[output].op.axis
        bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
        by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
        bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)
        kernel_scope = n  # this is the scope to attach global config inside this kernel

        s[output].reorder(n, bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
        fuse = s[output].fuse(yi, xi)
        s[output].vectorize(fuse)
        if (args[0] == 4 and args[1] == 112 and args[2] == 14 and args[4] == 224):
            s[output].unroll(fi)
        s[OL].compute_at(s[output], tx)

        # tile reduction axes
        n, f, y, x = s[OL].op.axis
        rc, ry, rx = s[OL].op.reduce_axis
        rco, rcm, rci = cfg['tile_rc'].apply(s, OL, rc)
        ryo, rym, ryi = cfg['tile_rx'].apply(s, OL, ry)
        rxo, rxm, rxi = cfg['tile_ry'].apply(s, OL, rx)
        s[OL].reorder(n, rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi, f, y, x)

        n, f, y, x = s[conv].op.axis
        s[conv].parallel(n)

        #print(tvm.lower(s, [data, kernel, conv], simple_mode=True))

        return s, [raw_data, kernel, conv]

def conv2d_turning_1x1(*args):
    global function
    ops, bufs = function(*args)
    data, kernel, conv = bufs
    s = tvm.create_schedule(conv.op)
    n, f, y, x = s[conv].op.axis
    rc, ry, rx = s[conv].op.reduce_axis

    if ry==y and rx==x:
        n, f, y, x = s[conv].op.axis
        rc, ry, rx = s[conv].op.reduce_axis
        fo,fi=s[conv].split(f,factor=16)
        xo,yo,xi,yi=s[conv].tile(y,x,x_factor=8,y_factor=8)

    else:
        cfg=autotvm.get.config()
        n, f, y, x = s[conv].op.axis
        rc, ry, rx = s[conv].op.reduce_axis
        
        factor_range = [2, 4, 8, 16, 32, 64]
        cfg.define_knob('tile_factor_x', factor_range)
        cfg.define_knob('tile_factor_y', factor_range)
        bx = cfg['tile_factor_x'].val
        by = cfg['tile_factor_y'].val

        fo,fi=s[conv].split(f,factor=bx)
        xo,yo,xi,yi=s[conv].tile(y,x,x_factor=by,y_factor=by)

    s[conv].reorder(n,fo,xo,yo,fi,xi,yi)
    s[conv].vectorize(yi)
    fused=s[conv].fuse(n,fo)
    s[conv].parallel(fused)

    return s, [data, kernel, conv]

def hello(p):
    os.kill(p.pid, signal.SIGKILL)

def auto_schedule(func, args):
    """Automatic scheduler
    
    Args:
    -----------------
    func: function object
        similar to batch_gemm function mentioned above
    args: tuple
        inputs to func
    -----------------
    Returns:
    s: tvm.schedule.Schedule
    bufs: list of tvm.tensor.Tensor
    """
    global function
    function = func

    ops, bufs = func(*args)
    #################################################
    # do some thing with `ops`, `bufs` and `args`
    # to analyze which schedule is appropriate
    if len(bufs[0].shape) == 3:
        # gemm
        log_name = "./auto_schedule/gemm.log"
        print("gemm")
        A = bufs[0]
        B = bufs[1]
        batch, N, L = A.shape
        batch, L, M = B.shape
        
        task = autotvm.task.create(gemm_tuning, args=(int(batch), int(N), int(L), int(M)), target='llvm')

    else:
        # conv2d
        log_name = "./auto_schedule/conv2d.log"
        print("conv2d")
        task = autotvm.task.create(conv2d_turning, args=(args), target='llvm')
        print(task.config_space)
        pass

    # logging config (for printing tuning log to the screen)
    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

    # There are two steps for measuring a config: build and run.
    # By default, we use all CPU cores to compile program. Then measure them sequentially.
    # We measure 5 times and take average to reduce variance.
    measure_option = autotvm.measure_option(
        builder='local',
        runner=autotvm.LocalRunner(timeout=200, number=2, repeat=2))
#        runner=autotvm.LocalRunner(repeat=3))
    # begin tuning, log records to file `matmul.log`
    # tuner = autotvm.tuner.RandomTuner(task)
    tuner = autotvm.tuner.RandomTuner(task)

  #  tuner.tune(n_trial=10,
  #          measure_option=measure_option,
  #          callbacks=[autotvm.callback.log_to_file(log_name)])

    p = Process(target=tuner.tune, 
        args=(1000, measure_option, None, [autotvm.callback.log_to_file(log_name)])
        )
    t = threading.Timer(300, hello, args=[p])

    t.start()
    p.start()
    p.join()

    # apply history best from log file
    with autotvm.apply_history_best(log_name):
        with tvm.target.create("llvm"):
            if log_name == "./auto_schedule/gemm.log":
                s, bufs = gemm_tuning(batch, N, L, M)
            elif log_name == "./auto_schedule/conv2d.log":
                s, bufs = conv2d_turning(*args)
                pass

    #################################################
    # perform real schedule according to 
    # decisions made above, using primitives 
    # such as split, reorder, parallel, unroll...
    
    #################################################
    # finally, remember to return these two results
    # we need `bufs` to build function via `tvm.build`
    return s, bufs
