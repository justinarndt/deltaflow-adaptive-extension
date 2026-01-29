`timescale 1ns / 1ps

module tb_dsb_latency;

    // Parameters
    parameter WIDTH = 16;
    parameter CLK_PERIOD = 3.33; // 300 MHz

    // Signals
    logic clk;
    logic rst_n;
    logic enable;
    logic signed [WIDTH-1:0] a_t;
    logic signed [WIDTH-1:0] coupling_force;
    logic signed [WIDTH-1:0] x_out;
    logic sign_bit;

    // Instantiate the Unit Under Test (UUT)
    dsb_pe #(
        .WIDTH(WIDTH)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .a_t(a_t),
        .coupling_force(coupling_force),
        .x_out(x_out),
        .sign_bit(sign_bit)
    );

    // Clock Generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Test Sequence
    initial begin
        $dumpfile("dsb_wave.vcd");
        $dumpvars(0, tb_dsb_latency);

        // 1. Initialize
        rst_n = 0;
        enable = 0;
        a_t = 0;
        coupling_force = 0;
        
        #(CLK_PERIOD * 5);
        rst_n = 1;
        
        // 2. Start Benchmark
        $display("=== RTL Latency Validation: dSB Processing Element ===");
        $display("Target Clock: 300 MHz (3.33ns period)");
        
        // Step 1: Inject a strong positive coupling (simulation of an error chain neighbor)
        // 1.5 in Q8.8 = 0x0180
        coupling_force = 16'h0180; 
        a_t = 0; // Start of bifurcation
        
        // Enable
        enable = 1;
        
        // Measure discrete steps
        #(CLK_PERIOD);
        $display("T=1 (3.3ns): x_out = %h (Should start moving positive)", x_out);
        
        #(CLK_PERIOD);
        $display("T=2 (6.6ns): x_out = %h", x_out);
        
        #(CLK_PERIOD);
        $display("T=3 (9.9ns): x_out = %h", x_out);
        
        #(CLK_PERIOD);
        $display("T=4 (13.3ns): x_out = %h", x_out);
        
        // Check Validity at 15ns mark
        if (x_out > 0) begin
            $display("[PASS] Signal Propagation < 15ns. State Updated Successfully.");
        end else begin
            $display("[FAIL] Logic too slow or incorrect.");
        end

        // 3. Ramp Bifurcation Parameter a(t)
        // In real hardware this ramps over ~100 cycles, we speed it up here
        repeat (10) begin
            #(CLK_PERIOD);
            a_t = a_t + 16'h0010; // Increment a(t)
            $display("Ramping a(t): %h | x_out: %h", a_t, x_out);
        end

        $finish;
    end

endmodule
