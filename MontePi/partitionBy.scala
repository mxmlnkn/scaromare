startSpark --time=04:00:00 --nodes=5 --partition=west --gres= --cpus-per-task=12
spark-shell --master=$MASTER_ADDRESS

import java.net.InetAddress
import org.apache.spark.Partitioner
val pairs = sc.parallelize( 0 to 4*12-1, 4*12-1 ).map( x=>(x,x) )


class ExactPartitioner[V]( partitions: Int, elements: Int) extends Partitioner {
    def numPartitions() : Int = partitions
    def getPartition(key: Any): Int = key.asInstanceOf[Int]
}


pairs.mapPartitionsWithIndex{(ip,it)=> it.map( (x) => { Thread.sleep(10); x+" : "+InetAddress.getLocalHost().getHostName()+" located in "+ip } )}.collect().foreach( println )

(0,0)   : taurusi3094 located in 0
(1,1)   : taurusi3092 located in 1
(2,2)   : taurusi3099 located in 2
(3,3)   : taurusi3100 located in 3
(4,4)   : taurusi3094 located in 4
(5,5)   : taurusi3092 located in 5
(6,6)   : taurusi3099 located in 6
(7,7)   : taurusi3100 located in 7
(8,8)   : taurusi3094 located in 8
(9,9)   : taurusi3092 located in 9
(10,10) : taurusi3099 located in 10
(11,11) : taurusi3100 located in 11
(12,12) : taurusi3094 located in 12
(13,13) : taurusi3092 located in 13
(14,14) : taurusi3099 located in 14
(15,15) : taurusi3100 located in 15
(16,16) : taurusi3094 located in 16
(17,17) : taurusi3092 located in 17
(18,18) : taurusi3099 located in 18
(19,19) : taurusi3100 located in 19
(20,20) : taurusi3094 located in 20
(21,21) : taurusi3092 located in 21
(22,22) : taurusi3099 located in 22
(23,23) : taurusi3100 located in 23
(24,24) : taurusi3094 located in 24
(25,25) : taurusi3092 located in 25
(26,26) : taurusi3099 located in 26
(27,27) : taurusi3100 located in 27
(28,28) : taurusi3094 located in 28
(29,29) : taurusi3092 located in 29
(30,30) : taurusi3099 located in 30
(31,31) : taurusi3100 located in 31
(32,32) : taurusi3094 located in 32
(33,33) : taurusi3092 located in 33
(34,34) : taurusi3099 located in 34
(35,35) : taurusi3100 located in 35
(36,36) : taurusi3094 located in 36
(37,37) : taurusi3092 located in 37
(38,38) : taurusi3099 located in 38
(39,39) : taurusi3100 located in 39
(40,40) : taurusi3094 located in 40
(41,41) : taurusi3092 located in 41
(42,42) : taurusi3099 located in 42
(43,43) : taurusi3100 located in 43
(44,44) : taurusi3094 located in 44
(45,45) : taurusi3092 located in 45
(46,46) : taurusi3099 located in 46
(47,47) : taurusi3099 located in 46  <- WHY?! Bug?


pairs.partitionBy( new ExactPartitioner(4*12,4*12) ).mapPartitionsWithIndex{(ip,it)=> it.map( (x) => { Thread.sleep(10); x+" : "+InetAddress.getLocalHost().getHostName()+" located in "+ip } )}.collect().foreach( println )

(0,0)   : taurusi3099 located in 0   ----------> 4 div 0  \
(1,1)   : taurusi3100 located in 1    +--------> 4 div 1  |
(2,2)   : taurusi3094 located in 2    |+-------> 4 div 2  | 4 mod 0 where 4 = nNodes
(3,3)   : taurusi3092 located in 3    ||+------> 4 div 3  |
(4,4)   : taurusi3099 located in 4   -+||+-----> 4 div 4  /
(5,5)   : taurusi3100 located in 5     |||
(6,6)   : taurusi3094 located in 6     |||
(7,7)   : taurusi3092 located in 7     |||
(8,8)   : taurusi3099 located in 8   --+||
(9,9)   : taurusi3100 located in 9      ||
(10,10) : taurusi3094 located in 10     ||
(11,11) : taurusi3092 located in 11     ||
(12,12) : taurusi3099 located in 12  ---+|
(13,13) : taurusi3100 located in 13      |
(14,14) : taurusi3094 located in 14      |
(15,15) : taurusi3092 located in 15      |
(16,16) : taurusi3099 located in 16  ----+
(17,17) : taurusi3100 located in 17
(18,18) : taurusi3094 located in 18
(19,19) : taurusi3092 located in 19
(20,20) : taurusi3099 located in 20
(21,21) : taurusi3100 located in 21
(22,22) : taurusi3094 located in 22
(23,23) : taurusi3092 located in 23
(24,24) : taurusi3099 located in 24
(25,25) : taurusi3100 located in 25
(26,26) : taurusi3094 located in 26
(27,27) : taurusi3092 located in 27
(28,28) : taurusi3099 located in 28
(29,29) : taurusi3100 located in 29
(30,30) : taurusi3094 located in 30
(31,31) : taurusi3092 located in 31
(32,32) : taurusi3099 located in 32
(33,33) : taurusi3100 located in 33
(34,34) : taurusi3094 located in 34
(35,35) : taurusi3092 located in 35
(36,36) : taurusi3099 located in 36
(37,37) : taurusi3100 located in 37
(38,38) : taurusi3094 located in 38
(39,39) : taurusi3092 located in 39
(40,40) : taurusi3099 located in 40
(41,41) : taurusi3100 located in 41
(42,42) : taurusi3094 located in 42
(43,43) : taurusi3092 located in 43
(44,44) : taurusi3099 located in 44
(45,45) : taurusi3100 located in 45
(46,46) : taurusi3094 located in 46
(47,47) : taurusi3092 located in 47


class ExactPartitioner[V]( partitions: Int, elements: Int) extends Partitioner {
    def numPartitions() : Int = partitions
    def getPartition(key: Any): Int = {
        val k = key.asInstanceOf[Int]
        val nCoresPerNode = 12
        val nNodes = ( partitions + nCoresPerNode - 1 ) / nCoresPerNode // ceilDiv
        ( k * nNodes  + k / nCoresPerNode ) % ( nNodes * nCoresPerNode )
    }
}
pairs.partitionBy( new ExactPartitioner(4*12,4*12) ).mapPartitionsWithIndex{(ip,it)=> it.map( (x) => { Thread.sleep(10); x+" : "+InetAddress.getLocalHost().getHostName()+" located in "+ip } )}.collect().foreach( println )

(0,0) : taurusi3099 located in 0
(12,12) : taurusi3092 located in 1
(24,24) : taurusi3100 located in 2
(36,36) : taurusi3094 located in 3
(1,1) : taurusi3099 located in 4
(13,13) : taurusi3092 located in 5
(25,25) : taurusi3100 located in 6
(37,37) : taurusi3094 located in 7
(2,2) : taurusi3099 located in 8
(14,14) : taurusi3092 located in 9
(26,26) : taurusi3100 located in 10
(38,38) : taurusi3094 located in 11
(3,3) : taurusi3099 located in 12
(15,15) : taurusi3092 located in 13
(27,27) : taurusi3100 located in 14
(39,39) : taurusi3094 located in 15
(4,4) : taurusi3099 located in 16
(16,16) : taurusi3092 located in 17
(28,28) : taurusi3100 located in 18
(40,40) : taurusi3094 located in 19
(5,5) : taurusi3099 located in 20
(17,17) : taurusi3092 located in 21
(29,29) : taurusi3100 located in 22
(41,41) : taurusi3094 located in 23
(6,6) : taurusi3099 located in 24
(18,18) : taurusi3092 located in 25
(30,30) : taurusi3100 located in 26
(42,42) : taurusi3094 located in 27
(7,7) : taurusi3099 located in 28
(19,19) : taurusi3092 located in 29
(31,31) : taurusi3100 located in 30
(43,43) : taurusi3094 located in 31
(8,8) : taurusi3099 located in 32
(20,20) : taurusi3092 located in 33
(32,32) : taurusi3100 located in 34
(44,44) : taurusi3094 located in 35
(9,9) : taurusi3099 located in 36
(21,21) : taurusi3092 located in 37
(33,33) : taurusi3100 located in 38
(45,45) : taurusi3094 located in 39
(10,10) : taurusi3099 located in 40
(22,22) : taurusi3092 located in 41
(34,34) : taurusi3100 located in 42
(46,46) : taurusi3094 located in 43
(11,11) : taurusi3099 located in 44
(23,23) : taurusi3092 located in 45
(35,35) : taurusi3100 located in 46
(47,47) : taurusi3094 located in 47

   |||||||||||||||||||||||||||
   |||||||||| Sort |||||||||||
   |||||||||||||||||||||||||||
   vvvvvvvvvvvvvvvvvvvvvvvvvvv

(0,0)   : taurusi3099 located in 0
(1,1)   : taurusi3099 located in 4
(2,2)   : taurusi3099 located in 8
(3,3)   : taurusi3099 located in 12
(4,4)   : taurusi3099 located in 16
(5,5)   : taurusi3099 located in 20
(6,6)   : taurusi3099 located in 24
(7,7)   : taurusi3099 located in 28
(8,8)   : taurusi3099 located in 32
(9,9)   : taurusi3099 located in 36
(10,10) : taurusi3099 located in 40
(11,11) : taurusi3099 located in 44

(12,12) : taurusi3092 located in 1
(13,13) : taurusi3092 located in 5
(14,14) : taurusi3092 located in 9
(15,15) : taurusi3092 located in 13
(16,16) : taurusi3092 located in 17
(17,17) : taurusi3092 located in 21
(18,18) : taurusi3092 located in 25
(19,19) : taurusi3092 located in 29
(20,20) : taurusi3092 located in 33
(21,21) : taurusi3092 located in 37
(22,22) : taurusi3092 located in 41
(23,23) : taurusi3092 located in 45

(24,24) : taurusi3100 located in 2
(25,25) : taurusi3100 located in 6
(26,26) : taurusi3100 located in 10
(27,27) : taurusi3100 located in 14
(28,28) : taurusi3100 located in 18
(29,29) : taurusi3100 located in 22
(30,30) : taurusi3100 located in 26
(31,31) : taurusi3100 located in 30
(32,32) : taurusi3100 located in 34
(33,33) : taurusi3100 located in 38
(34,34) : taurusi3100 located in 42
(35,35) : taurusi3100 located in 46

(36,36) : taurusi3094 located in 3
(37,37) : taurusi3094 located in 7
(38,38) : taurusi3094 located in 11
(39,39) : taurusi3094 located in 15
(40,40) : taurusi3094 located in 19
(41,41) : taurusi3094 located in 23
(42,42) : taurusi3094 located in 27
(43,43) : taurusi3094 located in 31
(44,44) : taurusi3094 located in 35
(45,45) : taurusi3094 located in 39
(46,46) : taurusi3094 located in 43
(47,47) : taurusi3094 located in 47

