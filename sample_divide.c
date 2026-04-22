/* Read two integers (e.g. from stdin) and print their quotient as a double. */

#include <stdio.h>

int main(void) {
    int a, b;
    printf("Enter two integers: ");
    fflush(stdout);
    if (scanf("%d %d", &a, &b) != 2) {
        fprintf(stderr, "Expected two integers\n");
        return 1;
    }
    if (b == 0) {
        fprintf(stderr, "Division by zero\n");
        return 1;
    }
    printf("quotient = %f\n", (double)a / (double)b);
    return 0;
}
