#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    int n;
    printf("Enter matrix size (1-50): ");
    scanf("%d", &n);
    
    if (n < 1 || n > 50) {
        printf("Invalid size!\n");
        return 1;
    }

    srand(time(NULL));
    FILE *fp = fopen("matrices.txt", "w");
    
    for (int m = 0; m < 12; m++) {
        int matrix[n][n];
        
        // Generate symmetric matrix with zero diagonal
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                if (i == j) {
                    matrix[i][j] = 0;
                } else {
                    int val = 1 + rand() % 99;
                    matrix[i][j] = val;
                    matrix[j][i] = val;
                }
            }
        }

        fprintf(fp, "%d: [", m);
        for (int i = 0; i < n; i++) {
            fprintf(fp, "[");
            for (int j = 0; j < n; j++) {
                fprintf(fp, "%d", matrix[i][j]);
                if (j < n-1) fprintf(fp, ", ");
            }
            fprintf(fp, "]%s\n", i < n-1 ? "," : "");
        }
        fprintf(fp, "],");
    }

    fclose(fp);
    printf("12 matrices written to matrices.txt\n");
    return 0;
}