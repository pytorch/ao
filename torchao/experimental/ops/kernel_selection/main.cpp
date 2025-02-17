#include <iostream>

#include <cpuinfo.h> 



/*

Kernel selection (for both linear and packing ops):

Goal: given the following inputs
1. packing format
2. cpuinfo
3. data size (m, n)

we need to select kernel function pointers or raise an exception.

Packing format can be: universal format, kleidiAI format, etc.
A given format can be supported by multiple kernels for multiple cpu_archs.

In terms of dispatch, we want to initalize cpuinfo once and reuse multiple times.
For a given packing format, we want to cache fn ptrs for the cpuinfo.  We can have multiple fn ptrs cached and they can be selected based on data size and core.


Packing format selection:

Given user itent (directly specifying packing format) or information like bitwidth/cpuinfo, we select a packign format.
For example, given bitwidth 4 and i8mm arch, we might select kliedi packing format.
Given bitwidth != 4, we select universal packing format.

Optionally, a user can directly specify a packing format for AOT packing.
*/

int main() {
  std::cout << "Hello, World!" << std::endl;

  cpuinfo_initialize();




  std::cout << "Uarch: " << cpuinfo_get_core(0)->uarch << std::endl;
  std::cout << "corecont: " << cpuinfo_package().core_count << std::endl;

  get_kernel_ptrs_for_universal_1x16() {

  }

  if (cpuinfo_has_arm_neon()) {
    if (cpuinfo_has_arm_i8mm()) {
      std::cout << "Arm i8mm" << std::endl;
    }
    
    if (cpuinfo_has_arm_neon_dot()) {
      std::cout << "Arm neon dot" << std::endl;
    }
    
    if (cpuinfo_has_arm_neon_fma()) {
      std::cout << "ARM neon fma" << std::endl;
    }
  } 
  else {
    throw std::runtime_error("Invalid archecture");
  }

  fptr[packing_format][core_id][m]


  
  if (cpuinfo_has_arm_neon()) {
    std::cout << "NEON is supported" << std::endl;
    std::cout << "curr uarch index: " << cpuinfo_get_current_uarch_index_with_default(182341) << std::endl;

    std::cout << "ARM DOT: " << cpuinfo_has_arm_neon_dot() << std::endl;




    auto current_core = cpuinfo_get_current_core();
    if (current_core != nullptr) {
      std::cout << "Current core proc start: " << current_core->processor_start;
    } else {
      std::cout << "Current core is null" << std::endl;
    }
  }


  return 0;
}
