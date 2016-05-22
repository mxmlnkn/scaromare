

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


# ToDo

- Benchmark Multi-GPU
  - Implement Multi-GPU using runAsync
    - try to convert MontePi.java to MontePi.scala so that it is easer to program
      - Try it out on singleNode/singleGpu/scala first
      
