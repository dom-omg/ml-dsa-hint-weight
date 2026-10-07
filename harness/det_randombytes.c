/*
 * Deterministic replacement for randombytes(): a SHAKE256 stream keyed by a
 * fixed label, the parameter set and a shard number, so that a data collection run can be
 * repeated bit for bit. Not for any use other than this experiment.
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "fips202.h"
#include "params.h"
#include "randombytes.h"

/* set by the harness before the first call; distinct shards give distinct keys */
uint8_t det_shard = 0;

static keccak_state state;
static int ready = 0;

void randombytes(uint8_t *out, size_t outlen) {
    if (!ready) {
        uint8_t seed[32] = "COBALT-PQC hint weight v1";
        seed[30] = det_shard;
        seed[31] = (uint8_t)DILITHIUM_MODE;
        shake256_init(&state);
        shake256_absorb(&state, seed, sizeof seed);
        shake256_finalize(&state);
        ready = 1;
    }
    shake256_squeeze(out, outlen, &state);
}
