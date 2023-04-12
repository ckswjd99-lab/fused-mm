TARGET_DIR := arm64

ifeq ($(ARCH), arm64)
	TARGET_DIR = arm64
endif

all:
	make -C ${TARGET_DIR}

clean:
	-make -C bench ${@F}
	-make -C ${TARGET_DIR} ${@F}