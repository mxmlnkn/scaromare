
# Starting MontePi Spark Multi GPU Test

    startSpark --time=15:00:00 --nodes=2 --partition=gpu2 --cpus-per-task=2 --gres='gpu:2'
    squeue -u $USER
         JOBID   PARTITION NAME     ST      TIME  NODES NODELIST(REASON)
         7660143 gpu2      start_sp R   14:49:09      2 taurusi[2093-2094]
    makeOptsSpark=(                                                      \
        "SPARK_ROOT=$HOME/spark-1.5.2-bin-hadoop2.6"                     \
        "SPARK_JAR=$SPARK_ROOT/lib/spark-assembly-1.5.2-hadoop2.6.0.jar" \
        "SPARKCORE_JAR=$SPARK_JAR"                                       \
        "SCALA_ROOT=$(dirname $(which scala))/../lib"                    \
    )
    ( cd "$HOME/scaromare/rootbeer1/" && ( cd csrc && ./compile_linux_x64 ) && ant clean && 'rm' -f dist/Rootbeer1.jar Rootbeer.jar && ant jar && ./pack-rootbeer ) && make clean && make -B && java -jar Count.jar
    folder=$HOME/scaromare/MontePi/multiNode/multiGpu/scala
    make -C "$folder" -B ${makeOptsSpark[@]} MontePi.jar
    sparkSubmit "$folder/MontePi.jar" 268435456 2 2 2>&1 | sed '/ INFO /d'


# Workings

    3. Rootbeer constructor -> CUDALoader:
          Load shared libraries with fixed absolute paths mostly ...
              m_libCudas.add("/usr/lib64/libcuda.so");
              m_libCudas.add("/usr/lib/x86_64-linux-gnu/libcudart.so.5.0");
              m_rootbeerRuntimes.add(RootbeerPaths.v().getRootbeerHome()+"rootbeer_x64.so.1");
              m_rootbeerCudas.add(RootbeerPaths.v().getRootbeerHome()+"rootbeer_cuda_x64.so.1");
    4. Rootbeer.getDevices
        CUDARuntime constructor -> loadGpuDevices -> CUDARuntime.c
            => Build list of cudaGetDeviceProperties. The C native (!) returns
               List<GpuDevice>
    5. GPUDevice.createContext -> CUDAContext -> native initializeDriver
            https://de.wikipedia.org/wiki/Java_Native_Interface
        ./csrc/org/trifort/rootbeer/runtime/CUDAContext.c:JNIEXPORT void JNICALL Java_org_trifort_rootbeer_runtime_CUDAContext_initializeDriver
            only saves function pointers to context.java
    6. Rootber.run
    6. 1. Context context = createDefaultContext(); // skipped for multi-GPU, instad user creates a context himself
          CUDAContext constructor
           - spawns worker thread (pool?) saved in m_exec and given to m_disruptor -> m_handler -> m_ringBuffer
           - calls CUDAContext.c : allocateNativeContext which allocates object of ContextState
               -> ContextState holds device and host pointers for serialized objects as well as the device,
               -> sets up kernel, copies data from and to device and launches kernel
    6. 2. context.setThreadConfig(thread_config);
    6. 3. context.setKernel( work.get(0) );
            sets m_kernelTemplate and m_compiledKernel
    6. 4. context.setUsingHandles(true); // activates some kind of additional memory ???
    6. 5. context.buildState();
             gpuEvent.setValue(GpuEventCommand.NATIVE_BUILD_STATE);
                 GpuEventHandler.onEvent
                     nativeBuildState( ..., gpuDevice.getDeviceId(), ... )
                         Java_org_trifort_rootbeer_runtime_CUDAContext_nativeBuildState in CUDAContext.c
          => "Sets specified GPU device, creates CUDA context, sets shared memory, configuration, kernel parameters and kernel configuration."
          Allocates FixedMemory objects m_classMemory, ... (see below)
          nativeBuildState handles the following data buffers:
            s->gpu_{object,handles,exceptions,class}_mem, gpu_heap_end
         e.g. with cuMemAlloc, cuMemFree
            heap_end and info_space are fixed 4 bytes long and very similar in content
            The size for the others is queried by FixedMemory.java:getSize through the JNI
    7. context.run(work)
        - context.runAsync(work)
           gpuEvent.setValue( GpuEventCommand.NATIVE_RUN_LIST );
             GpuEventHandler.onEvent
        - writeBlocksList( gpuEvent.getKernelList() );
            reset heaps
            m_compiledKernel.getSerializer( m_objectMemory, m_textureMemory ).writeStaticsToHeap()
              - doWriteStaticsToHeap <- defined using ByteCodeLanguage in VisitorWriteGenStatic.java makeMethod
              -
            then for each kernel in list do
                m_handlesMemory.writeRef( serializer.writeToHeap( kernel ) );
            m_objectMemory.align16()
        - runGpu
            cudaRun( m_nativeContext, m_objectMemory, !m_usingHandles ? 1 : 0, m_stats );
              Note that m_objectMemory is only used to get heapEnd of it, the address is only saved at nativeBuildState
              - cuMemcpyHtoD, cuLaunchGrid, cuMemcpyDtoH (see also 6. 5.)
                GPU: deviceGlobalFreePointer <-> info_space (heap_end >> 4)
                     s->gpu_object_mem       <-> s->cpu_object_mem_size
                     s->gpu_handles_mem      <-  s->cpu_handles_mem_size
                     s->gpu_exceptions_mem   <-> s->cpu_exceptions_mem
                     s->gpu_class_mem        <-  s->cpu_class_mem_size
                     s->gpu_heap_end         <-  heap_end_int
                     deviceMLocal            <-  hostMLocal (gpu_object_mem{,size}, gpu_class_mem)

        - readBlocksList(  gpuEvent.getKernelList() );
        - gpuEvent.getFuture().signal();

          private native void cudaRun(long nativeContext, Memory objectMem, int usingKernelTemplates, StatsRow stats);
        - synchronize using GpuFuture.take()


# Known Bugs

 -
        Exception in thread "main" java.lang.NoSuchMethodError: scala.runtime.ObjectRef.create(Ljava/lang/Object;)Lscala/runtime/ObjectRef;

    I had this error when trying to put MonteCarloPi.class in the jar which will be processed by Rootbeer instead of in the non-Rootbeer jar. After extracting the class to MonteCarloPi.class.old and comparing to the non Rootbeer compiled we get:
        4.6K MonteCarloPi.class
        3.9K MonteCarloPi.class.old
    Meaning the Rootbeer-processed class shrank!
        colordiff <(strings MonteCarloPi.class | sort) <(strings MonteCarloPi.classold | sort)
            5d4
            < b(Lorg/trifort/rootbeer/runtime/Rootbeer;ILjava/util/List<Lorg/trifort/rootbeer/runtime/Kernel;>;)V
            14d12
            < Creating Rootbeer Context...
            18d15
            <  from list of length
            22d18
            < Get device
            24d19
            < Get Device List
            38c33
            < (I)V
            ---
            > Jasmin
            47c42
            < java/util/List
            ---
            > java/util/List
            51d45
            < LineNumberTable
            68,69d61
            < MonteCarloPi constructor took
            < MonteCarloPi.java
            72a65
            > N-N+
            78d70
            < print
            87d78
            < 	Signature
            90d80
            < StackMapTable
            96c86
            < &Total iterations done by all kernels:
            ---
            > &Total iterations done by all kernels:
    => The contructor seems to have been stripped off! But other methods are still there Oo:
        strings MonteCarloPi.classold | grep 'Run a total of'
            Run a total of
        strings MonteCarloPi.class | grep 'Run a total of'
            Run a total of
    -> TODO: Look what Rootbeer does to make this shit happen ... (Note that a DummyFunction calling the constructor solves the problem also. So maybe it's just an optimization by Soot)

 - Trying to compile only MontePiKernel with Rootbeer and then merge MontePi into it results in:

    Exception in thread "main" java.lang.ClassCastException: MonteCarloPiKernel cannot be cast to org.trifort.rootbeer.runtime.CompiledKernel
        at org.trifort.rootbeer.runtime.CUDAContext.setKernel(CUDAContext.java:119)
        at org.trifort.rootbeer.runtime.Rootbeer.run(Rootbeer.java:88)
        at MonteCarloPi.calc(MonteCarloPi.java:56)
        at TestMonteCarloPi$.main(TestMonteCarloPi.scala:14)
        at TestMonteCarloPi.main(TestMonteCarloPi.scala)


    Exception in thread "main" java.lang.NoSuchMethodError: MonteCarloPi: method <init>()V not found
        at TestMonteCarloPi$.main(TestMonteCarloPi.scala:11)
        at TestMonteCarloPi.main(TestMonteCarloPi.scala)

 -
        java.lang.ClassCastException: MonteCarloPiKernel cannot be cast to org.trifort.rootbeer.runtime.CompiledKernel

    MonteCarloPiKernel.class which is in target jar wasn't sent through Rootbeer.jar, e.g. if it was accidentally added to MontePiCPU.jar

 -
       java.lang.ClassCastException: MonteCarloPiKernel cannot be cast to [J
        at MonteCarloPiKernel.org_trifort_readFromHeapRefFields_MonteCarloPiKernel0(Jasmin)
        at MonteCarloPiKernelSerializer.doReadFromHeap(Jasmin)
        at org.trifort.rootbeer.runtime.Serializer.readFromHeap(Serializer.java:155)
        at org.trifort.rootbeer.runtime.CUDAContext.readBlocksList(CUDAContext.java:452)
        at org.trifort.rootbeer.runtime.CUDAContext$GpuEventHandler.onEvent(CUDAContext.java:332)
        at org.trifort.rootbeer.runtime.CUDAContext$GpuEventHandler.onEvent(CUDAContext.java:308)
        at com.lmax.disruptor.BatchEventProcessor.run(BatchEventProcessor.java:128)
        at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1145)
        at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:615)
        at java.lang.Thread.run(Thread.java:724)

   ???

 -
       Exception in thread "main" java.lang.NoClassDefFoundError: org/trifort/rootbeer/runtime/Kernel

   It seems Rootbeer.jar wasn't merged into the fatjar or specified int he classpath. This may happen after the rootbeer commit where it doesn't do that automatically anymore.

 -
       16/05/23 12:01:39 ERROR FatalExceptionHandler: Exception processing: 1 org.trifort.rootbeer.runtime.GpuEvent@5a425c28
       java.lang.OutOfMemoryError: [FixedMemory.java]
           currentHeapEnd / bytes currently in use: 2670592 B
           Bytes requested to allocate            : 64 B
           total size available in FixedMemory    : 2670592 B
       (This happens if createContext(size) was called with an insufficient size, or CUDAContext.java:findMemorySize failed to determine the correct size automatically)
           at org.trifort.rootbeer.runtime.FixedMemory$MemPointer.mallocWithSize(FixedMemory.java:451)
           at org.trifort.rootbeer.runtime.FixedMemory.mallocWithSize(FixedMemory.java:306)
           at org.trifort.rootbeer.runtime.Serializer.checkWriteCache(Serializer.java:125)
           at org.trifort.rootbeer.runtime.Serializer.writeToHeap(Serializer.java:148)
           at org.trifort.rootbeer.runtime.Serializer.writeToHeap(Serializer.java:49)
           at org.trifort.rootbeer.runtime.CUDAContext.writeBlocksList(CUDAContext.java:473)
           at org.trifort.rootbeer.runtime.CUDAContext.access$1500(CUDAContext.java:19)
           at org.trifort.rootbeer.runtime.CUDAContext$GpuEventHandler.onEvent(CUDAContext.java:408)
           at org.trifort.rootbeer.runtime.CUDAContext$GpuEventHandler.onEvent(CUDAContext.java:369)
           at com.lmax.disruptor.BatchEventProcessor.run(BatchEventProcessor.java:128)
           at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1145)
           at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:615)
           at java.lang.Thread.run(Thread.java:724)

   => This seems to be a critical bug in the automatic memory management, especially `findMemorySize`. Try to use a manual estimate for the needed context size instead. Unfortunately I can't yet give a formula or even some estimates. It shouldn't be much more than all the member variables per Kernel times the kernel count.

 -
       context0.runAsync
       16/05/23 00:41:26 ERROR FatalExceptionHandler: Exception processing: 1 org.trifort.rootbeer.runtime.GpuEvent@59d04f4c
       java.lang.OutOfMemoryError: currentHeapEnd: 192 allocationSize: 213024 memorySize: 57156
           at org.trifort.rootbeer.runtime.FixedMemory$MemPointer.mallocWithSize(FixedMemory.java:445)
           at org.trifort.rootbeer.runtime.FixedMemory.mallocWithSize(FixedMemory.java:306)
           at org.trifort.rootbeer.runtime.Serializer.checkWriteCache(Serializer.java:88)
           at org.trifort.rootbeer.runtime.Serializer.writeToHeap(Serializer.java:110)
           at MonteCarloPiKernel.org_trifort_writeToHeapRefFields_MonteCarloPiKernel0(Jasmin)
           at MonteCarloPiKernelSerializer.doWriteToHeap(Jasmin)
           at org.trifort.rootbeer.runtime.Serializer.writeToHeap(Serializer.java:120)
           at org.trifort.rootbeer.runtime.Serializer.writeToHeap(Serializer.java:46)
           at org.trifort.rootbeer.runtime.CUDAContext.writeBlocksTemplate(CUDAContext.java:416)
           at org.trifort.rootbeer.runtime.CUDAContext.access$1200(CUDAContext.java:17)
           at org.trifort.rootbeer.runtime.CUDAContext$GpuEventHandler.onEvent(CUDAContext.java:376)
           at org.trifort.rootbeer.runtime.CUDAContext$GpuEventHandler.onEvent(CUDAContext.java:345)
           at com.lmax.disruptor.BatchEventProcessor.run(BatchEventProcessor.java:128)
           at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1145)
           at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:615)
           at java.lang.Thread.run(Thread.java:724)
       context1.runAsync

 -
       [MonteCarloPi.scala:<constructor>] Using the following GPUs : 0, 1,
       [MonteCarloPi.scala:<constructor>] MonteCarloPi constructor took 0.585073235 seconds
       1
       [MonteCarloPi.scala:calc] Running MonteCarlo on 2 GPUs with these maximum kernel configurations :
       [MonteCarloPi.scala:calc]     28672 28672
       [MonteCarloPi.scala:calc] with each these workloads / number of iterations :
       [MonteCarloPi.scala:calc] 13422 13421 [MonteCarloPi.scala:calc] These are the seed ranges for each partition of MonteCarloPi:
           17138123     160842860972783     321685704807444     482528548642104
           4611686018444526027     4611846861288360395     4612007704132194763     4612168546976030155
       [MonteCarloPi.scala:runOnDevice] [Host:taurusi2044,GPU:0] Total Thread Count = 28672, KernelConfig = (112,1,1) blocks with each (256,1,1) threads
       [MonteCarloPi.scala:runOnDevice] runOnDevice configuration took 0.633287853 seconds
       [MonteCarloPi.scala:runOnDevice] context.run( work ) took 1.08022E-4 seconds
       [MonteCarloPi.scala:runOnDevice] [Host:taurusi2044,GPU:1] Total Thread Count = 28672, KernelConfig = (112,1,1) blocks with each (256,1,1) threads
       [MonteCarloPi.scala:runOnDevice] runOnDevice configuration took 0.772282558 seconds
       [MonteCarloPi.scala:runOnDevice] context.run( work ) took 0.023839428 seconds
       [MonteCarloPi.scala:calc] Ran Kernels asynchronously on all GPUs. Took 2.223016115 seconds
       [MonteCarloPi.scala:calc] This is the list of kernel ranks for this host (one line per GPU) :
       [MonteCarloPi.scala:calc]     28672 28673 28674 28675 28676 28677 28678 28679 28680 28681 ...
       [MonteCarloPi.scala:calc] Taking from GpuFuture now (Wait for asynchronous tasks).
       16/05/23 12:03:58 ERROR Executor: Exception in task 0.0 in stage 8.0 (TID 5)
       java.lang.ClassCastException: MonteCarloPiKernel cannot be cast to [J
           at MonteCarloPiKernel.org_trifort_readFromHeapRefFields_MonteCarloPiKernel0(Jasmin)
           at MonteCarloPiKernelSerializer.doReadFromHeap(Jasmin)
           at org.trifort.rootbeer.runtime.Serializer.readFromHeap(Serializer.java:193)
           at org.trifort.rootbeer.runtime.CUDAContext.readBlocksList(CUDAContext.java:557)
           at org.trifort.rootbeer.runtime.CUDAContext$GpuEventHandler.onEvent(CUDAContext.java:410)
           at org.trifort.rootbeer.runtime.CUDAContext$GpuEventHandler.onEvent(CUDAContext.java:369)
           at com.lmax.disruptor.BatchEventProcessor.run(BatchEventProcessor.java:128)
           at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1145)
           at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:615)
           at java.lang.Thread.run(Thread.java:724)

    This error seems to alternate with the following error, which is more rare (same binary different runs):

       [MonteCarloPi.scala:<constructor>] Using the following GPUs : 0, 1,
       [MonteCarloPi.scala:<constructor>] MonteCarloPi constructor took 0.590675978 seconds
       1
       [MonteCarloPi.scala:calc] Running MonteCarlo on 2 GPUs with these maximum kernel configurations :
       [MonteCarloPi.scala:calc]     28672 28672
       [MonteCarloPi.scala:calc] with each these workloads / number of iterations :
       [MonteCarloPi.scala:calc] 13422 13421 [MonteCarloPi.scala:calc] These are the seed ranges for each partition of MonteCarloPi:
           17138123     160842860972783     321685704807444     482528548642104
           4611686018444526027     4611846861288360395     4612007704132194763     4612168546976030155
       [MonteCarloPi.scala:runOnDevice] [Host:taurusi2044,GPU:0] Total Thread Count = 28672, KernelConfig = (112,1,1) blocks with each (256,1,1) threads
       [MonteCarloPi.scala:runOnDevice] runOnDevice configuration took 0.625196456 seconds
       [MonteCarloPi.scala:runOnDevice] context.run( work ) took 7.6011E-5 seconds
       [MonteCarloPi.scala:runOnDevice] [Host:taurusi2044,GPU:1] Total Thread Count = 28672, KernelConfig = (112,1,1) blocks with each (256,1,1) threads
       [MonteCarloPi.scala:runOnDevice] runOnDevice configuration took 0.750625323 seconds
       [MonteCarloPi.scala:runOnDevice] context.run( work ) took 6.9326E-5 seconds
       [MonteCarloPi.scala:calc] Ran Kernels asynchronously on all GPUs. Took 2.134863127 seconds
       [MonteCarloPi.scala:calc] This is the list of kernel ranks for this host (one line per GPU) :
       [MonteCarloPi.scala:calc]     0 1 2 3 4 5 6 7 8 9 ...
       [MonteCarloPi.scala:calc]     28672 28673 28674 28675 28676 28677 28678 28679 28680 28681 ...
       [MonteCarloPi.scala:calc] Taking from GpuFuture now (Wait for asynchronous tasks).
       16/05/23 12:57:52 ERROR Executor: Exception in task 0.0 in stage 8.0 (TID 5)
       java.lang.NullPointerException
           at org.trifort.rootbeer.runtime.Serializer.checkWriteCache(Serializer.java:122)
           at org.trifort.rootbeer.runtime.Serializer.writeToHeap(Serializer.java:148)
           at MonteCarloPiKernel.org_trifort_writeToHeapRefFields_MonteCarloPiKernel0(Jasmin)
           at MonteCarloPiKernelSerializer.doWriteToHeap(Jasmin)
           at org.trifort.rootbeer.runtime.Serializer.writeToHeap(Serializer.java:158)
           at org.trifort.rootbeer.runtime.Serializer.writeToHeap(Serializer.java:49)
           at org.trifort.rootbeer.runtime.CUDAContext.writeBlocksList(CUDAContext.java:473)
           at org.trifort.rootbeer.runtime.CUDAContext.access$1500(CUDAContext.java:19)
           at org.trifort.rootbeer.runtime.CUDAContext$GpuEventHandler.onEvent(CUDAContext.java:408)
           at org.trifort.rootbeer.runtime.CUDAContext$GpuEventHandler.onEvent(CUDAContext.java:369)
           at com.lmax.disruptor.BatchEventProcessor.run(BatchEventProcessor.java:128)
           at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1145)
           at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:615)
           at java.lang.Thread.run(Thread.java:724)

    I have this problem with `multiGpuTestSpark`, but not with `multiGpuTest`.

    1st thought:

      - This seems to happen if there is still not enough memory allocated for the context, but where `java.lang.OutOfMemoryError: [FixedMemory.java]` doesn't detect this insufficient memory anymore. Increase memory by e.g. 1 MB: `device.createContext( ... + 1024*1024 );` => Even adding a padding of 64 MB won't work.

    new observation:

      - Only happens on Tesla K20xm reproducible 100%. Doesn't happen on K80 with the same binary!

    2nd thought:

      - Maybe the code is device specific? -> Full recompile doesn't help at all ...
      - try tackling the problem at the root -> force smaller kernel configurations:

        Working:
            Total Thread Count = 1024, KernelConfig = ( 4,1,1) blocks with each (256,1,1) threads
            Total Thread Count = 2048, KernelConfig = ( 8,1,1) blocks with each (256,1,1) threads
            Total Thread Count = 3072, KernelConfig = (12,1,1) blocks with each (256,1,1) threads
            Total Thread Count = 3584, KernelConfig = (14,1,1) blocks with each (256,1,1) threads
            Total Thread Count = 4096, KernelConfig = (16,1,1) blocks with each (256,1,1) threads
        Failed:
            Total Thread Count = 4608, KernelConfig = (18,1,1) blocks with each (256,1,1) threads
                assertion nIterations == nDiceRolls failes with nIterations =  12712,12088,(26843),26159,26189,26279,(26843),11955,12363 < 26843
                sometimes no error indicated with (26843)
            Total Thread Count = 5120, KernelConfig = (20,1,1) blocks with each (256,1,1) threads
                assertion nIterations == nDiceRolls failes with nIterations =  12313,11793 < 26843
                    possible error sources:
                        - wrong work distribution for odd multiples
                        - kernels didn't finish before memcpy device to host
                        - memory problem, meaning not all memory is being copied back from device to host
                        - all memory is being copied, but target buffer is too small?
        => The randomness seems to be proof for a concurrency problem. Furthermore it is weird, that the values seem to be distributed around two centers. That means the concurrency for the GPU devices seems to be discrete. I.e. at random results of only one or both GPUs and then again at random values smaller the allocated workload of:

        [MonteCarloPi.scala:calc] Running MonteCarlo on 2 GPUs with these maximum kernel configurations :
        [MonteCarloPi.scala:calc]     4608 4608
        [MonteCarloPi.scala:calc] with each these workloads / number of iterations :
        [MonteCarloPi.scala:calc] 13422 13421 [MonteCarloPi.scala:calc] These are the seed ranges for each partition of MonteCarloPi:
            17138123     1000799934331566     2001599851525010     3002399768718453
            4611686018444526027     4612686818361719243     4613687618278912459     4614688418196105675
        [MonteCarloPi.scala:runOnDevice] [Host:taurusi2044,GPU:0] Total Thread Count = 4608, KernelConfig = (18,1,1) blocks with each (256,1,1) threads
        [MonteCarloPi.scala:runOnDevice] runOnDevice configuration took 0.62157132 seconds
        [MonteCarloPi.scala:runOnDevice] context.run( work ) took 8.3022E-5 seconds
        [MonteCarloPi.scala:runOnDevice] [Host:taurusi2044,GPU:1] Total Thread Count = 4608, KernelConfig = (18,1,1) blocks with each (256,1,1) threads
        [MonteCarloPi.scala:runOnDevice] runOnDevice configuration took 0.747112951 seconds
        [MonteCarloPi.scala:runOnDevice] context.run( work ) took 6.0861E-5 seconds
        [MonteCarloPi.scala:calc] Ran Kernels asynchronously on all GPUs. Took 1.833641076 seconds
        [MonteCarloPi.scala:calc] This is the list of kernel ranks for this host (one line per GPU) :
        [MonteCarloPi.scala:calc]     0 1 2 3 4 5 6 7 8 9 ...
        [MonteCarloPi.scala:calc]     4608 4609 4610 4611 4612 4613 4614 4615 4616 4617 ...
        [MonteCarloPi.scala:calc] Taking from GpuFuture now (Wait for asynchronous tasks).
        [MonteCarloPi.scala:calc] Closing contexts now.
        [MonteCarloPi.scala:calc] synchronize (take) i.e. kernels took 1.1783939 seconds
        [MonteCarloPi.scala:calc] iterations actually done : 12363

    3rd though:

      - maybe it is a problem of how spark does parallelism? -> Try adding `cache()` where possible ... Unfortunately the problem seems to be inside a spark map function, so try to replace some `map` functions in `MonteCarlo.java` with for loops.

  -
        [MonteCarloPi.scala:calc] Running MonteCarlo on 2 GPUs with these maximum kernel configurations :
        [MonteCarloPi.scala:calc]     5120 5120
        [MonteCarloPi.scala:calc] with each these workloads / number of iterations :
        [MonteCarloPi.scala:calc] 13422 13421 [MonteCarloPi.scala:calc] These are the seed ranges for each partition of MonteCarloPi:
            17138123     900719942612222     1801439868086321     2702159793560421
            4611686018444526027     4612586738370000331     4613487458295474635     4614388178220948939
        [MonteCarloPi.scala:runOnDevice] [Host:taurusi2044,GPU:0] Total Thread Count = 5120, KernelConfig = (20,1,1) blocks with each (256,1,1) threads
        [MonteCarloPi.scala:runOnDevice] runOnDevice configuration took 0.625017593 seconds
        [MonteCarloPi.scala:runOnDevice] context.run( work ) took 1.11887E-4 seconds
        [MonteCarloPi.scala:runOnDevice] [Host:taurusi2044,GPU:1] Total Thread Count = 5120, KernelConfig = (20,1,1) blocks with each (256,1,1) threads
        [MonteCarloPi.scala:runOnDevice] runOnDevice configuration took 0.753767771 seconds
        [MonteCarloPi.scala:runOnDevice] context.run( work ) took 0.013884371 seconds
        [MonteCarloPi.scala:calc] Ran Kernels asynchronously on all GPUs. Took 1.91102629 seconds
        [MonteCarloPi.scala:calc] This is the list of kernel ranks for this host (one line per GPU) :
        [MonteCarloPi.scala:calc]     0 1 2 3 4 5 6 7 8 9 ...
        [MonteCarloPi.scala:calc]     5120 5121 5122 5123 5124 5125 5126 5127 5128 5129 ...
        [MonteCarloPi.scala:calc] Taking from GpuFuture now (Wait for asynchronous tasks).
        [MonteCarloPi.scala:calc] Closing contexts now.
        [MonteCarloPi.scala:calc] synchronize (take) i.e. kernels took 1.328204529 seconds
        [MonteCarloPi.scala:calc] iterations actually done : 11793
        16/05/23 13:39:08 ERROR Executor: Exception in task 0.0 in stage 8.0 (TID 5)
        java.lang.AssertionError: assertion failed
            at scala.Predef$.assert(Predef.scala:165)
            at MonteCarloPi.calc(MonteCarloPi.scala:262)
            at TestMonteCarloPi$$anonfun$4.apply(TestMonteCarloPi.scala:61)
            at TestMonteCarloPi$$anonfun$4.apply(TestMonteCarloPi.scala:59)
            at scala.collection.Iterator$$anon$11.next(Iterator.scala:328)
            at scala.collection.Iterator$class.foreach(Iterator.scala:727)
            at scala.collection.AbstractIterator.foreach(Iterator.scala:1157)
            at scala.collection.TraversableOnce$class.reduceLeft(TraversableOnce.scala:172)
            at scala.collection.AbstractIterator.reduceLeft(Iterator.scala:1157)
            at org.apache.spark.rdd.RDD$$anonfun$reduce$1$$anonfun$14.apply(RDD.scala:993)
            at org.apache.spark.rdd.RDD$$anonfun$reduce$1$$anonfun$14.apply(RDD.scala:991)
            at org.apache.spark.SparkContext$$anonfun$36.apply(SparkContext.scala:1943)
            at org.apache.spark.SparkContext$$anonfun$36.apply(SparkContext.scala:1943)
            at org.apache.spark.scheduler.ResultTask.runTask(ResultTask.scala:66)
            at org.apache.spark.scheduler.Task.run(Task.scala:88)
            at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:214)
            at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1145)
            at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:615)
            at java.lang.Thread.run(Thread.java:724)

    This happend on the K20x when debugging the `ClassCastException` error above while using `5*1024` threads. It means the number of Iterations done is not equal to the nDiceRolls requested, meaning something is wrong, e.g. kernels not being run, or the wrong amount of kernels being started.
      -> this doesn't really explain why it doesn't happen on K80, though

# Compiling on Taurus

Currently the example is using raw make, without CMake or a configure or similar, so you will have to configure some paths manually, e.g. on my Taurus:

    makeOptsSpark=(
        "SPARK_ROOT=$HOME/spark-1.5.2-bin-hadoop2.6"
        "SPARK_JAR=$HOME/spark-1.5.2-bin-hadoop2.6/lib/spark-assembly-1.5.2-hadoop2.6.0.jar"
        "SPARKCORE_JAR=$SPARK_JAR"
        "SCALA_ROOT=$(dirname $(which scala))/../lib"
    )

Also you need to have `zipmerge` and for Rootbeer Compilation `ant` in your `PATH` variable. E.g. I installed zipmerge manually to `~/programs/bin`.

Now you can compile e.g. the example using Spark und multiple GPUs with:

    make ${makeOptsSpark[@]} -C multiNode/multiGpu/scala/


# Benchmarking

## Weak Scaling

If a Spark instance is already running and the address of the master is saved in the variable `MASTER_ADDRESS`, then you can simply run:

    nGpusAvailable=32
    jarFile=$HOME/scaromare/MontePi/multiNode/multiGpu/scala/MontePi.jar
    fname=weak-scaling_$(date +%Y-%m-%d_%H-%M-%S)
    echo '' | tee -- "$fname-cpu.log" "$fname-gpu.log"
    printf "# Total Elements   Elements Per Thread    Partitions   Time / s   Calculated Pi\n" | tee -- "$fname-cpu.dat" "$fname-gpu.dat"
    for (( nPerPartition=2**37; nPerPartition<=2**41; nPerPartition*=4 )); do
      for (( iRepetition=0; iRepetition<3; iRepetition++ )); do
        # Spark + GPU
        for (( nPartitions=1; nPartitions <= $nGpusAvailable; nPartitions++ )) do
            sparkSubmit "$jarFile" $((nPerPartition*nPartitions)) $nPartitions 2>&1 |
                tee tmp.log |             # log file to extract times in seconds from
                tee -a "$fname-gpu.log" | # complete log file for human to view
                sed '/ INFO /d'           # delete numerous Spark messages
            seconds=$(sed -nr '
                s/.*Rolling the dice.*and took (.*) seconds.*/\1/p' tmp.log)
            pi=$(sed -nr 's/.*pi ~ ([0-9.]+).*/\1/p' tmp.log)

            printf "%16i %16i %4i %10.5f %s\n" $((nPerPartition*nPartitions)) \
                $nPerPartition $nPartitions $seconds $pi >> "$fname-gpu.dat"
        done
      done
    done

![Weak Scaling Plot (run times) with up to 32 K20x graphic cards](/MontePi/benchmark/weak-scaling-time-gpu.png?raw=true)

![ibid. Speedup](/MontePi/benchmark/weak-scaling-speedup-gpu.png?raw=true)

![ibid. Parallel Efficiency](/MontePi/benchmark/weak-scaling-efficiency-gpu.png?raw=true)


## Strong Scaling

    nGpusAvailable=32
    jarFile=$HOME/scaromare/MontePi/multiNode/multiGpu/scala/MontePi.jar
    fname=strong-scaling_$(date +%Y-%m-%d_%H-%M-%S)
    echo '' | tee -- "$fname-cpu.log" "$fname-gpu.log"
    printf "# Total Elements    Partitions   Time / s   Calculated Pi\n" | tee -- "$fname-cpu.dat" "$fname-gpu.dat"
    for (( n=2**37; n<=2**43; n*=4 )); do
      for (( iRepetition=0; iRepetition<5; iRepetition++ )); do
        # Spark + GPU
        for (( nPartitions=1; nPartitions <= $nGpusAvailable; nPartitions++ )) do
            sparkSubmit "$jarFile" $n $nPartitions 2>&1 |
                tee tmp.log |             # log file to extract times in seconds from
                tee -a "$fname-gpu.log" | # complete log file for human to view
                sed '/ INFO /d'           # delete numerous Spark messages
            seconds=$(sed -nr '
                s/.*Rolling the dice.*and took (.*) seconds.*/\1/p' tmp.log)
            pi=$(sed -nr 's/.*pi ~ ([0-9.]+).*/\1/p' tmp.log)

            printf "%16i %4i %10.5f %s\n" $n $nPartitions $seconds $pi >> "$fname-gpu.dat"
        done
      done
    done
    
![Strong Scaling Plot (run times) with up to 32 K20x graphic cards](/MontePi/benchmark/strong-scaling-time-logscale-gpu.png?raw=true)

![ibid. Speedup](/MontePi/benchmark/strong-scaling-speedup-gpu.png?raw=true)

![ibid. Parallel Efficiency](/MontePi/benchmark/strong-scaling-efficiency-gpu.png?raw=true)


## Plots

To make the plots `MontePi/benchmark/plot.py` can be used. Try also `python plot.py --help`.

    python plot.py -s strong-scaling_2016-08-04_03-40-03 -k weak-scaling_2016-08-03_22-52-00


## Profiling

    salloc --time=02:00:00 --nodes=1 --partition=gpu2-interactive --cpus-per-task=4 --gres='gpu:4' --mem-per-cpu=1000M
    make ${makeOptsSpark[@]} -C $HOME/scaromare/MontePi/singleNode/multiGpu/scala/ MontePi.jar
    srun java -jar $HOME/scaromare/MontePi/singleNode/multiGpu/scala/MontePi.jar 32435123451 4
    which java
        /sw/global/tools/java/jdk1.7.0_25/bin/java
    sbatch --time=00:30:00 --nodes=1 --partition=gpu2 --cpus-per-task=4 --gres='gpu:4' <<EOF
    #!/bin/bash
    nvprof --analysis-metrics --metrics all -o $HOME/scaromare/MontePi/profilingDataMultiGpuScala.nvp%p /sw/global/tools/java/jdk1.7.0_25/bin/java -jar $HOME/scaromare/MontePi/singleNode/multiGpu/scala/MontePi.jar $((4*14351234510)) 4
    EOF

Output:
    
    ==26439== NVPROF is profiling process 26439, command: /sw/global/tools/java/jdk1.7.0_25/bin/java -jar $HOME/scaromare/MontePi/singleNode/multiGpu/scala/MontePi.jar 1435123451 4
    [MonteCarloPi.scala:<constructor>] Using the following GPUs : 0 
    [MonteCarloPi.scala:<constructor>] MonteCarloPi constructor took 6.281092309 seconds
    0
    [MonteCarloPi.scala:runOnDevice] [Host:taurusi2051,GPU:0] Total Thread Count = 26624, KernelConfig = (104,1,1) blocks with each (256,1,1) threads
    [MonteCarloPi.scala:<constructor>] Using the following GPUs : 1 
    [MonteCarloPi.scala:<constructor>] MonteCarloPi constructor took 7.513381237 seconds
    0
    [MonteCarloPi.scala:<constructor>] Using the following GPUs : 2 
    [MonteCarloPi.scala:<constructor>] MonteCarloPi constructor took 7.538209838 seconds
    0
    [MonteCarloPi.scala:runOnDevice] runOnDevice configuration took 0.748453808 seconds
    [MonteCarloPi.scala:runOnDevice] context.run( work ) took 5.4038E-5 seconds
    [MonteCarloPi.scala:calc] Ran Kernels asynchronously on all GPUs. Took 1.265656465 seconds
    [MonteCarloPi.scala:calc] Taking from GpuFuture now (Wait for asynchronous tasks).
    [MonteCarloPi.scala:runOnDevice] [Host:taurusi2051,GPU:1] Total Thread Count = 26624, KernelConfig = (104,1,1) blocks with each (256,1,1) threads
    [MonteCarloPi.scala:runOnDevice] [Host:taurusi2051,GPU:2] Total Thread Count = 26624, KernelConfig = (104,1,1) blocks with each (256,1,1) threads
    [MonteCarloPi.scala:<constructor>] Using the following GPUs : 3 
    [MonteCarloPi.scala:<constructor>] MonteCarloPi constructor took 7.580257042 seconds
    0
    [MonteCarloPi.scala:runOnDevice] [Host:taurusi2051,GPU:3] Total Thread Count = 26624, KernelConfig = (104,1,1) blocks with each (256,1,1) threads

    [MonteCarloPi.scala:runOnDevice] runOnDevice configuration took 0.965182109 seconds
    [MonteCarloPi.scala:runOnDevice] context.run( work ) took 4.1006E-5 seconds
    [MonteCarloPi.scala:calc] Ran Kernels asynchronously on all GPUs. Took 1.017713559 seconds
    [MonteCarloPi.scala:calc] Taking from GpuFuture now (Wait for asynchronous tasks).
    [MonteCarloPi.scala:runOnDevice] runOnDevice configuration took 0.974981803 seconds
    [MonteCarloPi.scala:runOnDevice] context.run( work ) took 3.4606E-5 seconds
    [MonteCarloPi.scala:calc] Ran Kernels asynchronously on all GPUs. Took 1.004266163 seconds
    [MonteCarloPi.scala:calc] Taking from GpuFuture now (Wait for asynchronous tasks).
    [MonteCarloPi.scala:runOnDevice] runOnDevice configuration took 0.960264386 seconds
    [MonteCarloPi.scala:runOnDevice] context.run( work ) took 2.3148E-5 seconds
    [MonteCarloPi.scala:calc] Ran Kernels asynchronously on all GPUs. Took 0.976993471 seconds
    [MonteCarloPi.scala:calc] Taking from GpuFuture now (Wait for asynchronous tasks).



    [MonteCarloPi.scala:calc] synchronize (take) i.e. kernels took 1.673575898 seconds
    Join thread after 9.409900961 seconds
    [MonteCarloPi.scala:calc] synchronize (take) i.e. kernels took 1.296328742 seconds
    [MonteCarloPi.scala:calc] synchronize (take) i.e. kernels took 1.3390092 seconds
    [MonteCarloPi.scala:calc] synchronize (take) i.e. kernels took 1.326177838 seconds
    Join thread after 10.060772521 seconds
    Join thread after 10.25184158 seconds
    Join thread after 10.253567369 seconds
    Rolling the dice 1435123451 times resulted in pi ~ 3.141675075310326 and took 10.253777488 seconds
    ==26439== Generated result file: $HOME/scaromare/MontePi/profilingDataMultiGpuScala.nvp26439


# ToDo

- Benchmark Multi-GPU
  - Implement Multi-GPU using runAsync
    - try to convert MontePi.java to MontePi.scala so that it is easer to program
      - Try it out on singleNode/singleGpu/scala first

