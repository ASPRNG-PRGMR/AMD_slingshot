################################################################################
# Automatically-generated file. Do not edit!
################################################################################

SHELL = cmd.exe

# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: Arm Compiler'
	"D:/mspm0_files/ccs/tools/compiler/ti-cgt-armllvm_4.0.4.LTS/bin/tiarmclang.exe" -c @"device.opt"  -march=thumbv6m -mcpu=cortex-m0plus -mfloat-abi=soft -mlittle-endian -mthumb -O2 -I"D:/AMD_hardwre" -I"D:/AMD_hardwre/Debug" -I"D:/mspm0_files/mspm0_sdk_2_07_00_05/source/third_party/CMSIS/Core/Include" -I"D:/mspm0_files/mspm0_sdk_2_07_00_05/source" -gdwarf-3 -MMD -MP -MF"$(basename $(<F)).d_raw" -MT"$(@)"  $(GEN_OPTS__FLAG) -o"$@" "$<"
	@echo 'Finished building: "$<"'
	@echo ' '

build-759154055: ../empty.syscfg
	@echo 'Building file: "$<"'
	@echo 'Invoking: SysConfig'
	"D:/mspm0_files/ccs/utils/sysconfig_1.25.0/sysconfig_cli.bat" -s "D:/mspm0_files/mspm0_sdk_2_07_00_05/.metadata/product.json" --script "D:/AMD_hardwre/empty.syscfg" -o "." --compiler ticlang
	@echo 'Finished building: "$<"'
	@echo ' '

device_linker.cmd: build-759154055 ../empty.syscfg
device.opt: build-759154055
device.cmd.genlibs: build-759154055
ti_msp_dl_config.c: build-759154055
ti_msp_dl_config.h: build-759154055
Event.dot: build-759154055

%.o: ./%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: Arm Compiler'
	"D:/mspm0_files/ccs/tools/compiler/ti-cgt-armllvm_4.0.4.LTS/bin/tiarmclang.exe" -c @"device.opt"  -march=thumbv6m -mcpu=cortex-m0plus -mfloat-abi=soft -mlittle-endian -mthumb -O2 -I"D:/AMD_hardwre" -I"D:/AMD_hardwre/Debug" -I"D:/mspm0_files/mspm0_sdk_2_07_00_05/source/third_party/CMSIS/Core/Include" -I"D:/mspm0_files/mspm0_sdk_2_07_00_05/source" -gdwarf-3 -MMD -MP -MF"$(basename $(<F)).d_raw" -MT"$(@)"  $(GEN_OPTS__FLAG) -o"$@" "$<"
	@echo 'Finished building: "$<"'
	@echo ' '

startup_mspm0g350x_ticlang.o: D:/mspm0_files/mspm0_sdk_2_07_00_05/source/ti/devices/msp/m0p/startup_system_files/ticlang/startup_mspm0g350x_ticlang.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: Arm Compiler'
	"D:/mspm0_files/ccs/tools/compiler/ti-cgt-armllvm_4.0.4.LTS/bin/tiarmclang.exe" -c @"device.opt"  -march=thumbv6m -mcpu=cortex-m0plus -mfloat-abi=soft -mlittle-endian -mthumb -O2 -I"D:/AMD_hardwre" -I"D:/AMD_hardwre/Debug" -I"D:/mspm0_files/mspm0_sdk_2_07_00_05/source/third_party/CMSIS/Core/Include" -I"D:/mspm0_files/mspm0_sdk_2_07_00_05/source" -gdwarf-3 -MMD -MP -MF"$(basename $(<F)).d_raw" -MT"$(@)"  $(GEN_OPTS__FLAG) -o"$@" "$<"
	@echo 'Finished building: "$<"'
	@echo ' '


