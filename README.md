

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
