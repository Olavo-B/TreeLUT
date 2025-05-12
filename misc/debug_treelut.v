`timescale 1ns/1ps

// Primeiro modelo (implementação com expressões lógicas)
module class0_tree1_v1(input wire [185:0] i, output wire [2:0] o);

wire new_0_0, new_0_1, new_0_2, new_0_3,  new_0;
wire new_1_0,  new_1;
wire new_2_0, new_2_1,  new_2;
wire new_5_0,  new_5;
assign new_0_0 = i[86] & i[108] & i[139];
assign new_0_1 = i[86] & (~i[108]) & i[141];
assign new_0_2 = i[86] & (~i[108]) & (~i[141]);
assign new_0_3 = (~i[86]) & (~i[116]) & i[59];
assign new_0 = new_0_0 | new_0_1 | new_0_2 | new_0_3;
assign new_1_0 = (~i[86]) & (~i[116]) & (~i[59]);
assign new_1 = new_1_0;
assign new_2_0 = i[86] & i[108] & (~i[139]);
assign new_2_1 = (~i[86]) & i[116] & i[155];
assign new_2 = new_2_0 | new_2_1;
assign new_5_0 = (~i[86]) & i[116] & (~i[155]);
assign new_5 = new_5_0;
assign o = new_0 ? 0 : new_1 ? 1 : new_2 ? 2 : 5;

endmodule

// Segundo modelo (implementação com multiplexadores)
module class0_tree1_v2(input wire [185:0] i, output wire [2:0] o);

wire new_1;
wire new_2;
wire new_3;
wire new_4;
wire new_5;
wire new_6;
assign new_6 = i[59] ? 3'd0 : 3'd1;
assign new_5 = i[155] ? 3'd2 : 3'd5;
assign new_4 = i[141] ? 3'd0 : 3'd0;
assign new_3 = i[139] ? 3'd0 : 3'd2;
assign new_2 = i[116] ? new_5 : new_6;
assign new_1 = i[108] ? new_3 : new_4;
assign o = i[86] ? new_1 : new_2;


endmodule

// Testbench para comparar as duas implementações
module testbench_comparison();

// Declaração de sinais
reg [185:0] input_vector;
wire [2:0] output_v1;
wire [2:0] output_v2;
reg clk;
integer i, j, k, l, m, n;
integer mismatch_count;
integer test_count;

// Instanciação dos módulos a serem testados
class0_tree1_v1 uut1(
    .i(input_vector),
    .o(output_v1)
);

class0_tree1_v2 uut2(
    .i(input_vector),
    .o(output_v2)
);

// Inicialização
initial begin
    // Inicializa o vetor de entrada com zeros
    input_vector = 0;
    mismatch_count = 0;
    test_count = 0;
    clk = 0;
    
    
    // Executa o teste com todas as combinações dos bits relevantes
    for (i = 0; i < 2; i = i + 1) begin        // i[127]
        for (j = 0; j < 2; j = j + 1) begin    // i[94]
            for (k = 0; k < 2; k = k + 1) begin // i[92]
                for (l = 0; l < 2; l = l + 1) begin // i[60]
                    for (m = 0; m < 2; m = m + 1) begin // i[116]
                        for (n = 0; n < 2; n = n + 1) begin // i[95]
                            for (integer o = 0; o < 2; o = o + 1) begin // i[169]
                                #100;  // Espera para estabilizar
                                
                                // Configura os bits relevantes
                                input_vector[86] = i;
                                input_vector[108] = j;
                                input_vector[139] = k;
                                input_vector[141] = l;
                                input_vector[116] = m;
                                input_vector[155] = n;
                                input_vector[59] = o;
                                
                                #100;  // Espera para estabilizar
                                
                                // Verifica se as saídas são iguais
                                if (output_v1 !== output_v2) begin
                                    mismatch_count = mismatch_count + 1;
                                    $display("MISMATCH! 86:%d, 108:%d, 139:%d, 141:%d, 116:%d, 155:%d, 59:%d | output_v1=%d, output_v2=%d", 
                                              i, j, k, l, m, n, o, output_v1, output_v2);
                                end
                                
                                test_count = test_count + 1;
                            end
                        end
                    end
                end
            end
        end
    end
    
    // Exibe o resultado final
    $display("\n--- Test Summary ---");
    $display("Total tests: %d", test_count);
    $display("Mismatches: %d", mismatch_count);
    $display("Match percentage: %0.2f%%", (test_count - mismatch_count) * 100.0 / test_count);
    
    if (mismatch_count == 0)
        $display("TEST PASSED: Both implementations are functionally identical!");
    else
        $display("TEST FAILED: Implementations produce different results!");
    
    $finish;
end

endmodule
