#include <math.h>
#include <stdio.h>

int main(void) {
    unsigned long long prev = 1;
    unsigned long long curr = 1;

    for (int i = 1; i <= 42; i++) {
        double phi_approx = (double)curr / (double)prev;
        printf("step %2d: prev=%llu, curr=%llu  →  φ ≈ curr/prev = %.15f\n",
               i, prev, curr, phi_approx);

        unsigned long long next = prev + curr;
        prev = curr;
        curr = next;
    }

    printf("φ (closed form (1+√5)/2) = %.15f\n", (1.0 + sqrt(5.0)) / 2.0);
    return 0;
}
