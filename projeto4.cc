//ESTA CLASSE É IGUAL AO DO JUPYTER
//SERVE PARA SER MAIS FÁCIL DE ALTERAR E VISUALIZAR


#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

//Tentei usar __constant__, mas dava problema com a inicialização dos arrays
#define FEATURES 10
#define MAX_EXPR_LEN 200
#define MAX_DATA_ROWS 100000
#define MAX_FUNCTIONS 1000
#define THREADS_PER_BLOCK 256


//Não percebo o porquê mas isto é necessário na compilação
__device__ float parse_expr(const char* expr, int* pos, const float* features);
__device__ float parse_factor(const char* expr, int* pos, const float* features);



__host__ int read_data(float *data, float *targets) {
    FILE *file = fopen("data.csv", "r");
    if (file == NULL) {
        printf("Error opening data.csv\n");
        exit(1);
    }
    char line[1024];
    fgets(line, sizeof(line), file); // Skip header
    
    int num_rows = 0;
    while (fgets(line, sizeof(line), file) && num_rows < MAX_DATA_ROWS) {
        char *token = strtok(line, ",");
        token = strtok(NULL, ","); // Skip index
        
        for (int j = 0; j < FEATURES; j++) {
            if (token != NULL) {
                data[num_rows * FEATURES + j] = atof(token);
                token = strtok(NULL, ",");
            }
        }
        
        if (token != NULL) {
            targets[num_rows] = atof(token);
        }
        
        num_rows++;
    }
    fclose(file);
    return num_rows;
}


__host__ int read_functions(char* functions) {
    FILE* fp = fopen("functions.txt", "r");
    if (!fp) {
        printf("Error opening functions.txt\n");
        return 0;
    }
    
    int count = 0;
    char line[MAX_EXPR_LEN];
    
    while (fgets(line, sizeof(line), fp) && count < MAX_FUNCTIONS) {
        //Remove newline
        line[strcspn(line, "\n")] = 0;
        strncpy(functions + count * MAX_EXPR_LEN, line, MAX_EXPR_LEN - 1);
        functions[count * MAX_EXPR_LEN + MAX_EXPR_LEN - 1] = '\0';
        count++;
    }
    
    fclose(fp);
    return count;
}




__global__ void compute_squared_errors(const char* functions, const float* __restrict__ data, const float* __restrict__ targets, 
                                        int num_rows, int num_functions, float* __restrict__ errors) {
    int data_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int func_idx = blockIdx.y;
    
    if (data_idx >= num_rows || func_idx >= num_functions) {
        return;
    }
    
    //Menos problemas que arrays
    const char* expr = functions + func_idx * MAX_EXPR_LEN;
    const float* features = data + data_idx * FEATURES;
    
    int pos = 0;
    float predicted = parse_expr(expr, &pos, features);
    float error = predicted - targets[data_idx];
    
    //Erros são guardados já elevados ao quadrado,
	//distribui melhor a carga de trabalho
    errors[func_idx * num_rows + data_idx] = error * error;
}


__device__ void skip_whitespace(const char* expr, int* pos) {
    while (expr[*pos] == ' ') (*pos)++;
}



//Seccão de parsing feita com AI Claude, por mais que tentasse não consegui meter esta parte funcional sem AI
//Dificil de seguir logica de cabeça, aconselhado a acompanhar com desenho no papel
//Este método garante que a precedencia de parênteses é corretamente tratada

//Adições e subtrações
__device__ float parse_expr(const char* expr, int* pos, const float* features) {
    float result = parse_factor(expr, pos, features);
    
    while (1) {
        skip_whitespace(expr, pos);
        
        if (expr[*pos] == '+') {
            (*pos)++;
            result += parse_factor(expr, pos, features);
        } else if (expr[*pos] == '-') {
            (*pos)++;
            result -= parse_factor(expr, pos, features);
        } else {
            break;
        }
    }
    
    return result;
}

//Não existem multiplicações nem divisões no enunciado

//Parenteses, trignometria, e raizes (aka operaçoes com prioridade)
__device__ float parse_factor(const char* expr, int* pos, const float* features) {
    skip_whitespace(expr, pos);
    
    // Check for opening parenthesis
    if (expr[*pos] == '(') {
        (*pos)++; // consume '('
        float result = parse_expr(expr, pos, features);
        skip_whitespace(expr, pos);
        if (expr[*pos] == ')') {
			 (*pos)++; // consume ')'
		}
        return result;
    }
    
    if (expr[*pos] == 's' && expr[*pos+1] == 'i' && expr[*pos+2] == 'n' && expr[*pos+3] == 'f') {
        *pos += 4;
        skip_whitespace(expr, pos);
        if (expr[*pos] == '(') {
			 (*pos)++;
		}
        float arg = parse_expr(expr, pos, features);
        skip_whitespace(expr, pos);
        if (expr[*pos] == ')') {
			 (*pos)++;
		}
        return sinf(arg);
    }
    
    if (expr[*pos] == 'c' && expr[*pos+1] == 'o' && expr[*pos+2] == 's' && expr[*pos+3] == 'f') {
        *pos += 4;
        skip_whitespace(expr, pos);
        if (expr[*pos] == '(') {
			 (*pos)++;
		}
        float arg = parse_expr(expr, pos, features);
        skip_whitespace(expr, pos);
        if (expr[*pos] == ')') {
			 (*pos)++;
		}
        return cosf(arg);
    }
    
    if (expr[*pos] == 's' && expr[*pos+1] == 'q' && expr[*pos+2] == 'r' && expr[*pos+3] == 't' && expr[*pos+4] == 'f') {
        *pos += 5;
        skip_whitespace(expr, pos);
        if (expr[*pos] == '(') {
			 (*pos)++;
		}
        float arg = parse_expr(expr, pos, features);
        skip_whitespace(expr, pos);
        if (expr[*pos] == ')') {
			 (*pos)++;
		}
        return sqrtf(arg);
    }
    
    if (expr[*pos] == '_') {
        (*pos)++;
        char var = expr[*pos];
        (*pos)++;
        if (expr[*pos] == '_') {
			 (*pos)++;
		}
        
        if (var >= 'a' && var <= 'j') {
            return features[var - 'a'];
        }
    }
    
    return 0.0f;
}




__global__ void reduce_to_mse(const float* errors, int num_rows, int num_functions, float* mse) {
    

    __shared__ float shared_sum[THREADS_PER_BLOCK];
    
    int func_idx = blockIdx.x; 
    int tid = threadIdx.x;
    
    if (func_idx >= num_functions) {
        return;
    }
    
    //Usa stride
    float sum = 0.0f;
    for (int i = tid; i < num_rows; i += blockDim.x) {
        sum += errors[func_idx * num_rows + i];
    }
    
   
    shared_sum[tid] = sum;
    __syncthreads(); 
    
    // Tree-based reduction:
    //Ver gráfico no youtube: "CUDA Crash Course: Sum Reduction Part 1" aos aos 54 segundos
    for (int stride = blockDim.x / 2; stride > 0; stride = stride / 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();  //Necessário esperar para executar cada passo
    }
    
    
    if (tid == 0) {
        mse[func_idx] = shared_sum[0] / num_rows;
    }
}



int main() {
    printf("Teste GPU\n\n");

    //tempo total
    //Tive problemas a confirmar que o tempo está a ser medido corretamente
    //Aconselho o professor a verificar este ponto e ver se está tudo correto
    cudaEvent_t total_start, total_stop;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);
    cudaEventRecord(total_start);
    
    //Malloc para o CPU
    float *h_data = (float *)malloc(MAX_DATA_ROWS * FEATURES * sizeof(float));
    float *h_targets = (float *)malloc(MAX_DATA_ROWS * sizeof(float));
    char *h_functions = (char *)malloc(MAX_FUNCTIONS * MAX_EXPR_LEN * sizeof(char));
    float *h_mse = (float *)malloc(MAX_FUNCTIONS * sizeof(float));
    
    if (!h_data || !h_targets || !h_functions || !h_mse) {
        printf("Failed to allocate host memory\n");
        return 1;
    }
    
    //Ints necessário para quando o generate_inputs é alterado
    int num_rows = read_data(h_data, h_targets);
    int num_functions = read_functions(h_functions);
    
    //Malloc para a GPU
    float *d_data, *d_targets, *d_errors;
    char *d_functions;
    
    size_t data_size = num_rows * FEATURES * sizeof(float);
    size_t targets_size = num_rows * sizeof(float);
    size_t functions_size = num_functions * MAX_EXPR_LEN * sizeof(char);
    size_t errors_size = num_functions * num_rows * sizeof(float);
    size_t mse_size = num_functions * sizeof(float);
    
    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_targets, targets_size);
    cudaMalloc(&d_functions, functions_size);
    cudaMalloc(&d_errors, errors_size);
    //Metodo antigo de alocar o mse na GPU
    //cudaMalloc(&d_mse, mse_size);
    //(Nota 1) mse passou a ser allocado para memoria partilhada CPU/GPU
	//É o unico dado que tem que ser devolvido para a CPU
    float *mse;
    cudaMallocManaged(&mse, mse_size);
    
    
    //Copiar dados partilhados (usados por todas as funções) antes
    //Versão anterior incluia copia de funcoes, que agora é feita em streams
    //Como as funcoes precisam de todos os dados, estes têm que ser copiados antes
    cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, targets_size, cudaMemcpyHostToDevice);
    
    //Evento para medir o tempo de execução, prenda do Claude
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    //Criar múltiplas streams para task parallelism
    const int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    //x Para cada ponto, Y para cada funções
    dim3 threadsPerBlock(THREADS_PER_BLOCK, 1);
    //Implementacao antiga defenia blocks per grid aqui
    //Streams obrigam a definir mais tarde caso o numero de funcoes mude
    
    //Dividir funções entre streams
    int funcs_per_stream = (num_functions + NUM_STREAMS - 1) / NUM_STREAMS;

    
    //Nova versão com streams
    for (int s = 0; s < NUM_STREAMS; s++) {
        int start_func = s * funcs_per_stream;
        
        int end_func;
        if(start_func + funcs_per_stream < num_functions) {
            end_func = start_func + funcs_per_stream;
        } else {
            end_func = num_functions;
        }

        int funcs_in_stream = end_func - start_func;
        
        if (funcs_in_stream <= 0) continue;
        
        //Copiar apenas as funções necessárias para esta stream 
        //Foi aconselhado fazer async
        size_t func_chunk_size = funcs_in_stream * MAX_EXPR_LEN * sizeof(char);
        cudaMemcpyAsync(d_functions + start_func * MAX_EXPR_LEN, 
                        h_functions + start_func * MAX_EXPR_LEN,
                        func_chunk_size, 
                        cudaMemcpyHostToDevice, 
                        streams[s]);
        
        dim3 blocksPerGrid((num_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, funcs_in_stream);
        compute_squared_errors<<<blocksPerGrid, threadsPerBlock, 0, streams[s]>>>(
            d_functions + start_func * MAX_EXPR_LEN, 
            d_data, d_targets, 
            num_rows, funcs_in_stream, 
            d_errors + start_func * num_rows);
        
        reduce_to_mse<<<funcs_in_stream, THREADS_PER_BLOCK, 0, streams[s]>>>(
            d_errors + start_func * num_rows, 
            num_rows, funcs_in_stream, 
            mse + start_func);
    }
    
    //Sincronizar todas as streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    //Stop não para imediatamente a temporização
    //Em vez disso, mete o evento "paragem de cronometragem" em queue para quando todos os
    //kernels anteriores estiverem completos
    cudaEventRecord(stop);  
    
    //Metodo antigo de copiar o mse para a CPU
    //cudaMemcpy(h_mse, d_mse, mse_size, cudaMemcpyDeviceToHost);
    //(Nota 1) Se usar cudaMallocManaged, não é necessário este memcpy


    //Necessário esperar antes de ler
    cudaDeviceSynchronize();

    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    
    int min_idx = 0;
    float min_mse = mse[0];
    for (int i = 1; i < num_functions; i++) {
        if (mse[i] < min_mse) {
            min_mse = mse[i];
            min_idx = i;
        }
    }
    
    //stop tempo total
    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    
    float total_milliseconds = 0;
    cudaEventElapsedTime(&total_milliseconds, total_start, total_stop);

    //Alguma luta para perceber prints em C até hoje, resumo para a posteridade após pesquisa
    //% chama-se format specifier
    //Descobrir o nome dos % foi mais dificil que muitos projetos da faculdade
    //f no final é para indicar um float
    //.10 e .3 indicam o número de casas decimais após a vírgula
    //default é 6
    //Se tivesse um int em vez dum float, .10 indicaria o numero de digitos a serem mostrados
    //s é para strings, mas esse até eu sabia
    printf("Results\n");
    printf("First function MSE: %.10f %s\n", mse[0], h_functions);
    printf("Best function MSE: %.10f %s\n", min_mse, h_functions + min_idx * MAX_EXPR_LEN);
    printf("GPU kernel execution time: %.3f ms\n", milliseconds);
    printf("Total execution time (including data transfer): %.3f ms\n", total_milliseconds);

    
    
    cudaFree(d_data);
    cudaFree(d_targets);
    cudaFree(d_functions);
    cudaFree(d_errors);
    //Versão antiga
    //cudaFree(d_mse);
    
    //Pelos vistos é "necessário" destruir as streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);
    
    free(h_data);
    free(h_targets);
    free(h_functions);
    free(h_mse);
	
	cudaFree(mse);
    
    return 0;
}