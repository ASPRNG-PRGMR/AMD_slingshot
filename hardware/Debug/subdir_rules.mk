################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: Arm Compiler'
	"/home/noobiegg/ti/ccs2040/ccs/tools/compiler/ti-cgt-armllvm_4.0.4.LTS/bin/tiarmclang" -c @"device.opt"  -march=thumbv6m -mcpu=cortex-m0plus -mfloat-abi=soft -mlittle-endian -mthumb -O2 -I"/home/noobiegg/Documents/AMD_slingshot/hardware" -I"/home/noobiegg/Documents/AMD_slingshot/hardware/Debug" -I"/home/noobiegg/ti/mspm0_sdk_2_08_00_03/source/third_party/CMSIS/Core/Include" -I"/home/noobiegg/ti/mspm0_sdk_2_08_00_03/source" -gdwarf-3 -MMD -MP -MF"$(basename $(<F)).d_raw" -MT"$(@)"  $(GEN_OPTS__FLAG) -o"$@" "$(shell echo $<)"
	@echo 'Finished building: "$<"'
	@echo ' '

build-1601943456: ../empty.syscfg
	@echo 'Building file: "$<"'
	@echo 'Invoking: SysConfig'
	"/home/noobiegg/ti/ccs2040/ccs/utils/sysconfig_1.26.0/sysconfig_cli.sh" -s "/home/noobiegg/ti/mspm0_sdk_2_08_00_03/.metadata/product.json" --script "/home/noobiegg/Documents/AMD_slingshot/hardware/empty.syscfg" -o "." --compiler ticlang
	@echo 'Finished building: "$<"'
	@echo ' '

device_linker.cmd: build-1601943456 ../empty.syscfg
device.opt: build-1601943456
device.cmd.genlibs: build-1601943456
ti_msp_dl_config.c: build-1601943456
ti_msp_dl_config.h: build-1601943456
Event.dot: build-1601943456

%.o: ./%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: Arm Compiler'
	"/home/noobiegg/ti/ccs2040/ccs/tools/compiler/ti-cgt-armllvm_4.0.4.LTS/bin/tiarmclang" -c @"device.opt"  -march=thumbv6m -mcpu=cortex-m0plus -mfloat-abi=soft -mlittle-endian -mthumb -O2 -I"/home/noobiegg/Documents/AMD_slingshot/hardware" -I"/home/noobiegg/Documents/AMD_slingshot/hardware/Debug" -I"/home/noobiegg/ti/mspm0_sdk_2_08_00_03/source/third_party/CMSIS/Core/Include" -I"/home/noobiegg/ti/mspm0_sdk_2_08_00_03/source" -gdwarf-3 -MMD -MP -MF"$(basename $(<F)).d_raw" -MT"$(@)"  $(GEN_OPTS__FLAG) -o"$@" "$(shell echo $<)"
	@echo 'Finished building: "$<"'
	@echo ' '

startup_mspm0g350x_ticlang.o: /home/noobiegg/ti/mspm0_sdk_2_08_00_03/source/ti/devices/msp/m0p/startup_system_files/ticlang/startup_mspm0g350x_ticlang.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: Arm Compiler'
	"/home/noobiegg/ti/ccs2040/ccs/tools/compiler/ti-cgt-armllvm_4.0.4.LTS/bin/tiarmclang" -c @"device.opt"  -march=thumbv6m -mcpu=cortex-m0plus -mfloat-abi=soft -mlittle-endian -mthumb -O2 -I"/home/noobiegg/Documents/AMD_slingshot/hardware" -I"/home/noobiegg/Documents/AMD_slingshot/hardware/Debug" -I"/home/noobiegg/ti/mspm0_sdk_2_08_00_03/source/third_party/CMSIS/Core/Include" -I"/home/noobiegg/ti/mspm0_sdk_2_08_00_03/source" -gdwarf-3 -MMD -MP -MF"$(basename $(<F)).d_raw" -MT"$(@)"  $(GEN_OPTS__FLAG) -o"$@" "$(shell echo $<)"
	@echo 'Finished building: "$<"'
	@echo ' '


