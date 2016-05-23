

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

# ToDo

- Benchmark Multi-GPU
  - Implement Multi-GPU using runAsync
    - try to convert MontePi.java to MontePi.scala so that it is easer to program
      - Try it out on singleNode/singleGpu/scala first
      
