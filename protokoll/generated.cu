#define ARRAY_CHECKS

#ifndef NAN

#include <math_constants.h>

#define NAN CUDART_NAN

#endif

#ifndef INFINITY

#include <math_constants.h>

#define INFINITY CUDART_INF

#endif

#include <stdio.h>

__constant__ size_t m_Local[3];

__shared__ char m_shared[40960];

__device__
int getThreadId(){
  int blockSize = blockDim.x * blockDim.y * blockDim.z;
  
  int ret = (blockIdx.x * gridDim.y * blockSize) +
            (blockIdx.y * blockSize) +
            (threadIdx.x * blockDim.y * blockDim.z) +
            (threadIdx.y * blockDim.z) +
            (threadIdx.z);
  return ret;
}
__device__ clock_t global_now;

__device__ void 
java_lang_System_arraycopy(int src_handle, int srcPos, int dest_handle, int destPos, int length, int * exception);

__device__ int java_lang_Float_toString9_7_(float parameter0, int * exception);

__device__ int java_lang_Integer_toString9_5_(int parameter0, int * exception);

__device__ int java_lang_Object_initab850b60f96d11de8a390800200c9a660_(int * exception);

__device__ void java_lang_Object_initab850b60f96d11de8a390800200c9a66_body0_(int thisref, int * exception);

__device__ int java_lang_Double_toString9_8_(double parameter0, int * exception);

__device__ int invoke_java_lang_String_hashCode5_(int thisref, int * exception);

__device__ int invoke_java_lang_Object_hashCode(int thisref, int * exception);

__device__ int java_lang_Error_initab850b60f96d11de8a390800200c9a660_(int * exception);

__device__ void java_lang_Error_initab850b60f96d11de8a390800200c9a66_body0_(int thisref, int * exception);

__device__ int org_trifort_rootbeer_runtimegpu_GpuException_initab850b60f96d11de8a390800200c9a660_(int * exception);

__device__ void org_trifort_rootbeer_runtimegpu_GpuException_initab850b60f96d11de8a390800200c9a66_body0_(int thisref, int * exception);

__device__ int java_lang_Boolean_toString9_1_(char parameter0, int * exception);

__device__ int java_lang_Character_toString9_3_(char parameter0, int * exception);

__device__ int org_trifort_rootbeer_runtimegpu_GpuException_arrayOutOfBounds(int parameter0, int parameter1, int parameter2, int * exception);

__device__ int java_lang_Throwable_initab850b60f96d11de8a390800200c9a660_(int * exception);

__device__ void java_lang_Throwable_initab850b60f96d11de8a390800200c9a66_body0_(int thisref, int * exception);

__device__ char char__array_get(int thisref, int parameter0, int * exception);

__device__ void char__array_set(int thisref, int parameter0, char parameter1, int * exception);

__device__ int char__array_new(int size, int * exception);

__device__ int char__array_new_multi_array(int dim0, int * exception);

__device__ int java_lang_String__array_get(int thisref, int parameter0, int * exception);

__device__ void java_lang_String__array_set(int thisref, int parameter0, int parameter1, int * exception);

__device__ int java_lang_String__array_new(int size, int * exception);

__device__ int java_lang_String__array_new_multi_array(int dim0, int * exception);

__device__ void MonteCarloPiKernel_gpuMethod0_(int thisref, int * exception);

__device__ long long instance_getter_org_trifort_rootbeer_runtime_GpuStopwatch_m_start(int thisref, int * exception);

__device__ void instance_setter_org_trifort_rootbeer_runtime_GpuStopwatch_m_start(int thisref, long long parameter0, int * exception);

__device__ long long instance_getter_MonteCarloPiKernel_mRandomSeed(int thisref, int * exception);

__device__ void instance_setter_MonteCarloPiKernel_mRandomSeed(int thisref, long long parameter0, int * exception);

__device__ int instance_getter_MonteCarloPiKernel_mnHits(int thisref, int * exception);

__device__ void instance_setter_MonteCarloPiKernel_mnHits(int thisref, int parameter0, int * exception);

__device__ int instance_getter_org_trifort_rootbeer_runtimegpu_GpuException_m_arrayIndex(int thisref, int * exception);

__device__ void instance_setter_org_trifort_rootbeer_runtimegpu_GpuException_m_arrayIndex(int thisref, int parameter0, int * exception);

__device__ long long instance_getter_org_trifort_rootbeer_runtime_GpuStopwatch_m_stop(int thisref, int * exception);

__device__ void instance_setter_org_trifort_rootbeer_runtime_GpuStopwatch_m_stop(int thisref, long long parameter0, int * exception);

__device__ int instance_getter_java_lang_Throwable_suppressedExceptions(int thisref, int * exception);

__device__ void instance_setter_java_lang_Throwable_suppressedExceptions(int thisref, int parameter0, int * exception);

__device__ int instance_getter_java_lang_Integer_value(int thisref, int * exception);

__device__ void instance_setter_java_lang_Integer_value(int thisref, int parameter0, int * exception);

__device__ int instance_getter_org_trifort_rootbeer_runtimegpu_GpuException_m_arrayLength(int thisref, int * exception);

__device__ void instance_setter_org_trifort_rootbeer_runtimegpu_GpuException_m_arrayLength(int thisref, int parameter0, int * exception);

__device__ int static_getter_java_lang_Throwable_SUPPRESSED_SENTINEL(int * exception);

__device__ void static_setter_java_lang_Throwable_SUPPRESSED_SENTINEL(int parameter0, int * expcetion);

__device__ int instance_getter_java_lang_AbstractStringBuilder_count(int thisref, int * exception);

__device__ void instance_setter_java_lang_AbstractStringBuilder_count(int thisref, int parameter0, int * exception);

__device__ int instance_getter_java_lang_AbstractStringBuilder_value(int thisref, int * exception);

__device__ void instance_setter_java_lang_AbstractStringBuilder_value(int thisref, int parameter0, int * exception);

__device__ long long instance_getter_MonteCarloPiKernel_mnDiceRolls(int thisref, int * exception);

__device__ void instance_setter_MonteCarloPiKernel_mnDiceRolls(int thisref, long long parameter0, int * exception);

__device__ int instance_getter_org_trifort_rootbeer_runtimegpu_GpuException_m_array(int thisref, int * exception);

__device__ void instance_setter_org_trifort_rootbeer_runtimegpu_GpuException_m_array(int thisref, int parameter0, int * exception);

__device__ int instance_getter_java_lang_Throwable_cause(int thisref, int * exception);

__device__ void instance_setter_java_lang_Throwable_cause(int thisref, int parameter0, int * exception);

__device__ char static_getter_MonteCarloPiKernel_$assertionsDisabled(int * exception);

__device__ void static_setter_MonteCarloPiKernel_$assertionsDisabled(char parameter0, int * expcetion);

__device__ int instance_getter_java_lang_Class_name(int thisref, int * exception);

__device__ void instance_setter_java_lang_Class_name(int thisref, int parameter0, int * exception);

__device__ float instance_getter_java_lang_Float_value(int thisref, int * exception);

__device__ void instance_setter_java_lang_Float_value(int thisref, float parameter0, int * exception);

__device__ double instance_getter_java_lang_Double_value(int thisref, int * exception);

__device__ void instance_setter_java_lang_Double_value(int thisref, double parameter0, int * exception);

__device__ int instance_getter_java_lang_Throwable_stackTrace(int thisref, int * exception);

__device__ void instance_setter_java_lang_Throwable_stackTrace(int thisref, int parameter0, int * exception);

__device__ int static_getter_java_lang_Throwable_UNASSIGNED_STACK(int * exception);

__device__ void static_setter_java_lang_Throwable_UNASSIGNED_STACK(int parameter0, int * expcetion);

__device__ char instance_getter_java_lang_Boolean_value(int thisref, int * exception);

__device__ void instance_setter_java_lang_Boolean_value(int thisref, char parameter0, int * exception);

__device__ long long instance_getter_java_lang_Long_value(int thisref, int * exception);

__device__ void instance_setter_java_lang_Long_value(int thisref, long long parameter0, int * exception);

__device__ int instance_getter_MonteCarloPiKernel_miLinearThreadId(int thisref, int * exception);

__device__ void instance_setter_MonteCarloPiKernel_miLinearThreadId(int thisref, int parameter0, int * exception);

__device__ int org_trifort_rootbeer_runtime_Sentinal_initab850b60f96d11de8a390800200c9a660_(int * exception);

__device__ void org_trifort_rootbeer_runtime_Sentinal_initab850b60f96d11de8a390800200c9a66_body0_(int thisref, int * exception);

__device__ int java_lang_Long_toString9_6_(long long parameter0, int * exception);

__device__ int java_lang_StackTraceElement__array_get(int thisref, int parameter0, int * exception);

__device__ void java_lang_StackTraceElement__array_set(int thisref, int parameter0, int parameter1, int * exception);

__device__ int java_lang_StackTraceElement__array_new(int size, int * exception);

__device__ int java_lang_StackTraceElement__array_new_multi_array(int dim0, int * exception);

__device__ int java_lang_AssertionError_initab850b60f96d11de8a390800200c9a660_(int * exception);

__device__ void java_lang_AssertionError_initab850b60f96d11de8a390800200c9a66_body0_(int thisref, int * exception);

__device__ int java_lang_Object_hashCode(int thisref, int * exception);

__device__ long long long__array_get(int thisref, int parameter0, int * exception);

__device__ void long__array_set(int thisref, int parameter0, long long parameter1, int * exception);

__device__ int long__array_new(int size, int * exception);

__device__ int long__array_new_multi_array(int dim0, int * exception);

__device__ int java_lang_String_initab850b60f96d11de8a390800200c9a660_a13_(int parameter0, int * exception);

__device__ void java_lang_String_initab850b60f96d11de8a390800200c9a66_body0_a13_(int thisref, int parameter0, int * exception);

__device__ int java_lang_String_hashCode5_(int thisref, int * exception);

#define GC_OBJ_TYPE_COUNT char

#define GC_OBJ_TYPE_COLOR char

#define GC_OBJ_TYPE_TYPE int

#define GC_OBJ_TYPE_CTOR_USED char

#define GC_OBJ_TYPE_SIZE int

#define COLOR_GREY 0

#define COLOR_BLACK 1

#define COLOR_WHITE 2

#define OBJECT_HEADER_POSITION_GC_COUNT         0

#define OBJECT_HEADER_POSITION_GC_COLOR         1

#define OBJECT_HEADER_POSITION_CTOR_USED        3

#define OBJECT_HEADER_POSITION_CLASS_NUMBER     4

#define OBJECT_HEADER_POSITION_OBJECT_SIZE      8

#define OBJECT_HEADER_POSITION_MONITOR          16

#define ARRAY_HEADER_SIZE 32

#define INT_SIZE 4

#define LONG_SIZE 8

#define FLOAT_SIZE 4

#define DOUBLE_SIZE 8

__device__ void org_trifort_gc_collect();

__device__ void org_trifort_gc_assign(int * lhs, int rhs);

__device__  char * org_trifort_gc_deref(int handle);

__device__ int org_trifort_gc_malloc(int size);

__device__ int org_trifort_gc_malloc_no_fail(int size);

__device__ int org_trifort_classConstant(int type_num);

__device__ long long java_lang_System_nanoTime(int * exception);

#define CACHE_SIZE_BYTES 32

#define CACHE_SIZE_INTS (CACHE_SIZE_BYTES / sizeof(int))

#define CACHE_ENTRY_SIZE 4

__device__ int org_trifort_getint( char * buffer, int pos){
  return *(( int *) &buffer[pos]);
}
__device__ char
org_trifort_cmp(long long lhs, long long rhs){
  if(lhs > rhs)
    return 1;
  if(lhs < rhs)
    return -1;
  return 0;
}
__device__ char
org_trifort_cmpg(double lhs, double rhs){
  if(lhs > rhs)
    return 1;
  if(lhs < rhs)
    return -1;
  if(lhs == rhs)
    return 0;
  return 1;
}
__device__ void
org_trifort_gc_set_count( char * mem_loc, GC_OBJ_TYPE_COUNT value){
  mem_loc[0] = value;
}
__device__ void
org_trifort_gc_set_color( char * mem_loc, GC_OBJ_TYPE_COLOR value){
  mem_loc += sizeof(GC_OBJ_TYPE_COUNT);
  mem_loc[0] = value;
}
__device__ void
org_trifort_gc_init_monitor( char * mem_loc){
  int * addr;
  mem_loc += 16;
  addr = (int *) mem_loc;
  *addr = -1;
}
__device__ void
org_trifort_gc_set_type( char * mem_loc, GC_OBJ_TYPE_TYPE value){
  mem_loc += sizeof(GC_OBJ_TYPE_COUNT) + sizeof(GC_OBJ_TYPE_COLOR) + sizeof(char) +
    sizeof(GC_OBJ_TYPE_CTOR_USED);
  *(( GC_OBJ_TYPE_TYPE *) &mem_loc[0]) = value;
}
__device__ void
org_trifort_gc_set_ctor_used( char * mem_loc, GC_OBJ_TYPE_CTOR_USED value){
  mem_loc += sizeof(GC_OBJ_TYPE_COUNT) + sizeof(GC_OBJ_TYPE_COLOR) + sizeof(char);
  mem_loc[0] = value;
}
__device__ void
org_trifort_gc_set_size( char * mem_loc, GC_OBJ_TYPE_SIZE value){
  mem_loc += sizeof(GC_OBJ_TYPE_COUNT) + sizeof(GC_OBJ_TYPE_COLOR) + sizeof(char) + 
    sizeof(GC_OBJ_TYPE_CTOR_USED) + sizeof(GC_OBJ_TYPE_TYPE);
  *(( GC_OBJ_TYPE_SIZE *) &mem_loc[0]) = value;
}
__device__ void java_lang_String_initab850b60f96d11de8a390800200c9a66_body0_a12_(int thisref, int parameter0, int * exception);

__device__ int 
char__array_new(int size, int * exception);

__device__ void 
char__array_set(int thisref, int parameter0, char parameter1, int * exception);

__device__ void
org_trifort_gc_assign(int * lhs_ptr, int rhs){
  *lhs_ptr = rhs;
}
__device__ int java_lang_StackTraceElement__array_get(int thisref, int parameter0, int * exception);

__device__ void java_lang_StackTraceElement__array_set(int thisref, int parameter0, int parameter1, int * exception);

__device__ int java_lang_StackTraceElement__array_new(int size, int * exception);

__device__ int java_lang_StackTraceElement_initab850b60f96d11de8a390800200c9a660_3_3_3_4_(int parameter0, int parameter1, int parameter2, int parameter3, int * exception);

__device__ void instance_setter_java_lang_RuntimeException_stackDepth(int thisref, int parameter0);

__device__ int instance_getter_java_lang_RuntimeException_stackDepth(int thisref);

__device__ int java_lang_StackTraceElement__array_get(int thisref, int parameter0, int * exception);

__device__ int instance_getter_java_lang_Throwable_stackTrace(int thisref, int * exception);

__device__ void instance_setter_java_lang_Throwable_stackTrace(int thisref, int parameter0, int * exception);

__device__ int java_lang_Throwable_fillInStackTrace(int thisref, int * exception){
  
  
  return thisref;
}
__device__ void instance_setter_java_lang_Throwable_cause(int thisref, int parameter0, int * exception);

__device__ void instance_setter_java_lang_Throwable_detailMessage(int thisref, int parameter0, int * exception);

__device__ void instance_setter_java_lang_Throwable_stackDepth(int thisref, int parameter0, int * exception);

__device__ void java_lang_VirtualMachineError_initab850b60f96d11de8a390800200c9a66_body0_(int thisref, int * exception);

__device__ int java_lang_AssertionError_initab850b60f96d11de8a390800200c9a660_(int * exception){
int r0 = -1;
int thisref;
 char * thisref_deref;
thisref = -1;
org_trifort_gc_assign(&thisref, org_trifort_gc_malloc(32));
if(thisref == -1){
  *exception = 24212;
  return -1;
}
thisref_deref = org_trifort_gc_deref(thisref);
org_trifort_gc_set_count(thisref_deref, 0);
org_trifort_gc_set_color(thisref_deref, COLOR_GREY);
org_trifort_gc_set_type(thisref_deref, 20257);
org_trifort_gc_set_ctor_used(thisref_deref, 1);
org_trifort_gc_set_size(thisref_deref, 32);
org_trifort_gc_init_monitor(thisref_deref);
instance_setter_java_lang_Throwable_cause(thisref, -1, exception);
instance_setter_java_lang_Throwable_stackTrace(thisref, -1, exception);
instance_setter_java_lang_Throwable_suppressedExceptions(thisref, -1, exception);
 r0  =  thisref ;
java_lang_Error_initab850b60f96d11de8a390800200c9a66_body0_(thisref, exception);
return r0;
  return 0;
}
__device__ int org_trifort_rootbeer_runtimegpu_GpuException_arrayOutOfBounds(int parameter0, int parameter1, int parameter2, int * exception){
int i0 = 0;
int i1 = 0;
int i2 = 0;
int $r0 = -1;
int r1 = -1;
 i0  =  parameter0 ;
 i1  =  parameter1 ;
 i2  =  parameter2 ;
 $r0  =  -1 ;
org_trifort_gc_assign (
&$r0, org_trifort_rootbeer_runtimegpu_GpuException_initab850b60f96d11de8a390800200c9a660_(exception));
if(*exception != 0) {
 
return 0; }
 r1  =  $r0 ;
instance_setter_org_trifort_rootbeer_runtimegpu_GpuException_m_arrayLength(r1,  i2 , exception);
if(*exception != 0) {
 
return 0; }
instance_setter_org_trifort_rootbeer_runtimegpu_GpuException_m_arrayIndex(r1,  i0 , exception);
if(*exception != 0) {
 
return 0; }
instance_setter_org_trifort_rootbeer_runtimegpu_GpuException_m_array(r1,  i1 , exception);
if(*exception != 0) {
 
return 0; }
return  r1 ;
  return 0;
}
__device__ void long__array_set(int thisref, int parameter0, long long parameter1, int * exception){
int length;
 char * thisref_deref;
  if(thisref == -1){
    *exception = 24334;
    return;
  }
thisref_deref = org_trifort_gc_deref(thisref);
length = org_trifort_getint(thisref_deref, 12);
if(parameter0 < 0 || parameter0 >= length){
  *exception = org_trifort_rootbeer_runtimegpu_GpuException_arrayOutOfBounds(parameter0, thisref, length, exception);  return;
}
*(( long long *) &thisref_deref[32+(parameter0*8)]) = parameter1;
}
__device__ int org_trifort_rootbeer_runtimegpu_GpuException_initab850b60f96d11de8a390800200c9a660_(int * exception){
int r0 = -1;
int thisref;
 char * thisref_deref;
thisref = -1;
org_trifort_gc_assign(&thisref, org_trifort_gc_malloc(48));
if(thisref == -1){
  *exception = 24212;
  return -1;
}
thisref_deref = org_trifort_gc_deref(thisref);
org_trifort_gc_set_count(thisref_deref, 0);
org_trifort_gc_set_color(thisref_deref, COLOR_GREY);
org_trifort_gc_set_type(thisref_deref, 3875);
org_trifort_gc_set_ctor_used(thisref_deref, 1);
org_trifort_gc_set_size(thisref_deref, 48);
org_trifort_gc_init_monitor(thisref_deref);
instance_setter_org_trifort_rootbeer_runtimegpu_GpuException_m_array(thisref, 0, exception);
instance_setter_org_trifort_rootbeer_runtimegpu_GpuException_m_arrayIndex(thisref, 0, exception);
instance_setter_org_trifort_rootbeer_runtimegpu_GpuException_m_arrayLength(thisref, 0, exception);
 r0  =  thisref ;
return r0;
  return 0;
}
__device__ void java_lang_Error_initab850b60f96d11de8a390800200c9a66_body0_(int thisref, int * exception){
int r0 = -1;
 r0  =  thisref ;
java_lang_Throwable_initab850b60f96d11de8a390800200c9a66_body0_(thisref, exception);
return;
}
__device__ void java_lang_Throwable_initab850b60f96d11de8a390800200c9a66_body0_(int thisref, int * exception){
int r0 = -1;
int $r1 = -1;
int $r2 = -1;
 r0  =  thisref ;
instance_setter_java_lang_Throwable_cause(r0,  r0 , exception);
if(*exception != 0) {
 
return ; }
 $r1  = static_getter_java_lang_Throwable_UNASSIGNED_STACK(exception);
instance_setter_java_lang_Throwable_stackTrace(r0,  $r1 , exception);
if(*exception != 0) {
 
return ; }
 $r2  = static_getter_java_lang_Throwable_SUPPRESSED_SENTINEL(exception);
instance_setter_java_lang_Throwable_suppressedExceptions(r0,  $r2 , exception);
if(*exception != 0) {
 
return ; }
java_lang_Throwable_fillInStackTrace(r0,
 exception);
if(*exception != 0) {
 
return ; }
return;
}
__device__ void MonteCarloPiKernel_gpuMethod0_(int thisref, int * exception){
int r0 = -1;
int i0 = 0;
int i1 = 0;
long long l2 = 0;
int i3 = 0;
float f0 = 0;
float f1 = 0;
long long $l4 = 0;
char $z0 = 0;
long long $l5 = 0;
char $b6 = 0;
int $r1 = -1;
long long $l7 = 0;
long long $l8 = 0;
long long $l9 = 0;
long long $l10 = 0;
int i11 = 0;
float $f2 = 0;
long long $l12 = 0;
long long $l13 = 0;
long long $l14 = 0;
float $f3 = 0;
float $f4 = 0;
float $f5 = 0;
float $f6 = 0;
double $d0 = 0;
char $b15 = 0;
int $r2 = -1;
int $i16 = 0;
 r0  =  thisref ;
 $l4  = instance_getter_MonteCarloPiKernel_mRandomSeed(r0, exception);
if(*exception != 0) {
 
return ; }
 i0  = (int)  $l4 ;
 $z0  = static_getter_MonteCarloPiKernel_$assertionsDisabled(exception);
if ( $z0  !=  0   ) goto label0;
 $l5  = instance_getter_MonteCarloPiKernel_mnDiceRolls(r0, exception);
if(*exception != 0) {
 
return ; }
 $b6  = org_trifort_cmp( $l5 ,  2147483647L );
if ( $b6  <=  0   ) goto label0;
 $r1  =  -1 ;
org_trifort_gc_assign (
&$r1, java_lang_AssertionError_initab850b60f96d11de8a390800200c9a660_(exception));
if(*exception != 0) {
 
return ; }
 *exception =  $r1 ;
return;
label0:
 $l7  = instance_getter_MonteCarloPiKernel_mnDiceRolls(r0, exception);
if(*exception != 0) {
 
return ; }
 i1  = (int)  $l7 ;
 l2  =  0L ;
 i3  =  0 ;
label4:
if ( i3  >=  i1   ) goto label2;
 $l8  = (long long)  i0 ;
 $l9  =  950706376L  *  $l8  ;
 $l10  =  $l9  %  2147483647L  ;
 i11  = (int)  $l10 ;
 $f2  = (float)  i11 ;
 f0  =  $f2  /  2.14748365E9F  ;
 $l12  = (long long)  i11 ;
 $l13  =  950706376L  *  $l12  ;
 $l14  =  $l13  %  2147483647L  ;
 i0  = (int)  $l14 ;
 $f3  = (float)  i0 ;
 f1  =  $f3  /  2.14748365E9F  ;
 $f4  =  f0  *  f0  ;
 $f5  =  f1  *  f1  ;
 $f6  =  $f4  +  $f5  ;
 $d0  = (double)  $f6 ;
 $b15  = org_trifort_cmpg((double) $d0 , (double) 1.0 );
if ( $b15  >=  0   ) goto label3;
 l2  =  l2  +  1L  ;
label3:
 i3  =  i3  +  1  ;
goto label4;
label2:
 $r2  = instance_getter_MonteCarloPiKernel_mnHits(r0, exception);
if(*exception != 0) {
 
return ; }
 $i16  = instance_getter_MonteCarloPiKernel_miLinearThreadId(r0, exception);
if(*exception != 0) {
 
return ; }
long__array_set($r2, $i16,  l2 , exception);
if(*exception != 0) {
 
return ; }
return;
}
__device__ void instance_setter_org_trifort_rootbeer_runtimegpu_GpuException_m_array(int thisref, int parameter0, int * exception){
 char * thisref_deref;
if(thisref == -1){
  *exception = 24334;
  return;
}
thisref_deref = org_trifort_gc_deref(thisref);
*(( int *) &thisref_deref[32]) = parameter0;
}
__device__ char static_getter_MonteCarloPiKernel_$assertionsDisabled(int * exception){
 char * thisref_deref = org_trifort_gc_deref(0);
return *(( char *) &thisref_deref[8]);
}
__device__ int instance_getter_MonteCarloPiKernel_miLinearThreadId(int thisref, int * exception){
 char * thisref_deref;
if(thisref == -1){
  *exception = 24334;
  return 0;
}
thisref_deref = org_trifort_gc_deref(thisref);
return *(( int *) &thisref_deref[56]);
}
__device__ int static_getter_java_lang_Throwable_UNASSIGNED_STACK(int * exception){
 char * thisref_deref = org_trifort_gc_deref(0);
return *(( int *) &thisref_deref[0]);
}
__device__ long long instance_getter_MonteCarloPiKernel_mnDiceRolls(int thisref, int * exception){
 char * thisref_deref;
if(thisref == -1){
  *exception = 24334;
  return 0;
}
thisref_deref = org_trifort_gc_deref(thisref);
return *(( long long *) &thisref_deref[48]);
}
__device__ void instance_setter_org_trifort_rootbeer_runtimegpu_GpuException_m_arrayLength(int thisref, int parameter0, int * exception){
 char * thisref_deref;
if(thisref == -1){
  *exception = 24334;
  return;
}
thisref_deref = org_trifort_gc_deref(thisref);
*(( int *) &thisref_deref[40]) = parameter0;
}
__device__ int instance_getter_MonteCarloPiKernel_mnHits(int thisref, int * exception){
 char * thisref_deref;
if(thisref == -1){
  *exception = 24334;
  return 0;
}
thisref_deref = org_trifort_gc_deref(thisref);
return *(( int *) &thisref_deref[32]);
}
__device__ void instance_setter_org_trifort_rootbeer_runtimegpu_GpuException_m_arrayIndex(int thisref, int parameter0, int * exception){
 char * thisref_deref;
if(thisref == -1){
  *exception = 24334;
  return;
}
thisref_deref = org_trifort_gc_deref(thisref);
*(( int *) &thisref_deref[36]) = parameter0;
}
__device__ long long instance_getter_MonteCarloPiKernel_mRandomSeed(int thisref, int * exception){
 char * thisref_deref;
if(thisref == -1){
  *exception = 24334;
  return 0;
}
thisref_deref = org_trifort_gc_deref(thisref);
return *(( long long *) &thisref_deref[40]);
}
__device__ void instance_setter_java_lang_Throwable_cause(int thisref, int parameter0, int * exception){
 char * thisref_deref;
if(thisref == -1){
  *exception = 24334;
  return;
}
thisref_deref = org_trifort_gc_deref(thisref);
*(( int *) &thisref_deref[32]) = parameter0;
}
__device__ int static_getter_java_lang_Throwable_SUPPRESSED_SENTINEL(int * exception){
 char * thisref_deref = org_trifort_gc_deref(0);
return *(( int *) &thisref_deref[4]);
}
__device__ void instance_setter_java_lang_Throwable_suppressedExceptions(int thisref, int parameter0, int * exception){
 char * thisref_deref;
if(thisref == -1){
  *exception = 24334;
  return;
}
thisref_deref = org_trifort_gc_deref(thisref);
*(( int *) &thisref_deref[40]) = parameter0;
}
__device__ void instance_setter_java_lang_Throwable_stackTrace(int thisref, int parameter0, int * exception){
 char * thisref_deref;
if(thisref == -1){
  *exception = 24334;
  return;
}
thisref_deref = org_trifort_gc_deref(thisref);
*(( int *) &thisref_deref[36]) = parameter0;
}
__device__  char *
org_trifort_gc_deref(int handle){
  char * data_arr = (char * ) m_Local[0];
  long long lhandle = handle;
  lhandle = lhandle << 4;
  return &data_arr[lhandle];
}
__device__ int
org_trifort_gc_malloc(int size){
  int space_size = (int) m_Local[1];
  int ret = org_trifort_gc_malloc_no_fail(size);
  int end = ret + ((size + 16) >> 4);
  if(end >= space_size){
    return -1;
  }
  return ret;
}
__device__ int global_free_pointer; 

__device__ int
org_trifort_gc_malloc_no_fail(int size){
  if(size % 16 != 0){
    size += (16 - (size % 16));
  }
  size >>= 4;
  int ret;
  ret = atomicAdd(&global_free_pointer, size);
  return ret;
}
__global__ void entry(int * handles, int * exceptions, int numThreads, 
  int usingKernelTemplates){
  int totalThreadId = getThreadId();
  if(totalThreadId < numThreads){
    int exception = 0; 
    int handle;
    if(usingKernelTemplates){
      handle = handles[0];
    }
 else {
      handle = handles[totalThreadId];
    }
    MonteCarloPiKernel_gpuMethod0_(handle, &exception);  
    if(1){
      exceptions[totalThreadId] = exception;
    }
  }
}

