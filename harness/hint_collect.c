/*
 * Hint-weight collection on the pq-crystals reference implementation.
 *
 * For each key: the Euclidean norm of every polynomial of t0, then the
 * per-polynomial hint weights of n_sigs ordinary signatures, then the same
 * for n_sigs signatures kept only when the total weight lies in [lo, hi]
 * (bounded-weight signing), with the number of signing runs this took.
 *
 * Output (stdout), one record per line:
 *   N,<key>,<norm_0>,...,<norm_{K-1}>
 *   S,<key>,<w_0>,...,<w_{K-1}>
 *   B,<key>,<w_0>,...,<w_{K-1}>
 *   C,<key>,<signing runs used for the B records>
 *
 * Randomness is a SHAKE256 stream (det_randombytes.c), so a run is
 * reproducible from its arguments.
 *
 * Usage: ./hint_collect_<mode> <n_keys> <n_sigs> <lo> <hi> <shard>
 *
 * Shards are independent runs (own random stream); key ids are
 * shard * n_keys + index, so shards can run in parallel and be concatenated.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "params.h"
#include "sign.h"
#include "packing.h"
#include "polyvec.h"

extern uint8_t det_shard;

static void hint_weights(const uint8_t *sig, int w[K]) {
    const uint8_t *marks = sig + CTILDEBYTES + L * POLYZ_PACKEDBYTES + OMEGA;
    int prev = 0;
    for (int k = 0; k < K; k++) {
        w[k] = (int)marks[k] - prev;
        prev = (int)marks[k];
    }
}

static void print_weights(char tag, int key, const int w[K]) {
    printf("%c,%d", tag, key);
    for (int k = 0; k < K; k++) printf(",%d", w[k]);
    printf("\n");
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <n_keys> <n_sigs> <lo> <hi> <shard>\n", argv[0]);
        return 2;
    }
    const int n_keys = atoi(argv[1]);
    const int n_sigs = atoi(argv[2]);
    const int lo = atoi(argv[3]);
    const int hi = atoi(argv[4]);
    const int shard = atoi(argv[5]);
    if (n_keys <= 0 || n_sigs <= 0 || lo < 0 || hi < lo || hi > OMEGA || shard < 0 || shard > 255) {
        fprintf(stderr, "bad arguments\n");
        return 2;
    }

    uint8_t pk[CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[CRYPTO_SECRETKEYBYTES];
    uint8_t sig[CRYPTO_BYTES];
    uint8_t msg[16];
    size_t siglen;
    int w[K];

    det_shard = (uint8_t)shard;
    for (int idx = 0; idx < n_keys; idx++) {
        const int ki = shard * n_keys + idx;
        if (crypto_sign_keypair(pk, sk) != 0) return 1;

        uint8_t rho[SEEDBYTES], tr[TRBYTES], key[SEEDBYTES];
        polyveck t0, s2;
        polyvecl s1;
        unpack_sk(rho, tr, key, &t0, &s1, &s2, sk);
        printf("N,%d", ki);
        for (int k = 0; k < K; k++) {
            double acc = 0.0;
            for (int j = 0; j < N; j++) {
                double c = (double)t0.vec[k].coeffs[j];
                acc += c * c;
            }
            printf(",%.3f", sqrt(acc));
        }
        printf("\n");

        uint64_t ctr = 0;
        for (int si = 0; si < n_sigs; si++, ctr++) {
            memset(msg, 0, sizeof msg);
            memcpy(msg, &ki, sizeof ki);
            memcpy(msg + 8, &ctr, sizeof ctr);
            if (crypto_sign_signature(sig, &siglen, msg, sizeof msg, NULL, 0, sk) != 0) return 1;
            /* the first signatures of every key are checked with the reference verifier */
            if (si < 20 && crypto_sign_verify(sig, siglen, msg, sizeof msg, NULL, 0, pk) != 0) {
                fprintf(stderr, "verification failed: key %d sig %d\n", ki, si);
                return 1;
            }
            hint_weights(sig, w);
            print_weights('S', ki, w);
        }

        long runs = 0;
        int kept = 0;
        while (kept < n_sigs) {
            memset(msg, 0, sizeof msg);
            memcpy(msg, &ki, sizeof ki);
            memcpy(msg + 8, &ctr, sizeof ctr);
            ctr++;
            runs++;
            if (crypto_sign_signature(sig, &siglen, msg, sizeof msg, NULL, 0, sk) != 0) return 1;
            hint_weights(sig, w);
            int total = 0;
            for (int k = 0; k < K; k++) total += w[k];
            if (total < lo || total > hi) continue;
            print_weights('B', ki, w);
            kept++;
        }
        printf("C,%d,%ld\n", ki, runs);
        fflush(stdout);
        fprintf(stderr, "mode %d shard %d key %d/%d\n", DILITHIUM_MODE, shard, idx + 1, n_keys);
    }
    return 0;
}
