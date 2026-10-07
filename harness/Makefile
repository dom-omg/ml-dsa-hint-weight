# Hint-weight collection on the pq-crystals reference implementation.
#
#   git clone https://github.com/pq-crystals/dilithium dilithium-ref
#   git -C dilithium-ref checkout 6e00625
#   make DILITHIUM_REF=dilithium-ref/ref

DILITHIUM_REF ?= dilithium-ref/ref
CC     ?= cc
CFLAGS ?= -O2 -Wall -Wextra

REF = sign.c packing.c polyvec.c poly.c ntt.c reduce.c rounding.c fips202.c symmetric-shake.c
SRCS = hint_collect.c det_randombytes.c $(addprefix $(DILITHIUM_REF)/,$(REF))

.PHONY: all clean
all: hint_collect_2 hint_collect_3 hint_collect_5

hint_collect_%: $(SRCS)
	$(CC) $(CFLAGS) -DDILITHIUM_MODE=$* -I$(DILITHIUM_REF) $(SRCS) -lm -o $@

clean:
	rm -f hint_collect_2 hint_collect_3 hint_collect_5
