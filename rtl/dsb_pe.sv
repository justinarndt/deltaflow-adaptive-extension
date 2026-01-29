`timescale 1ns / 1ps

module dsb_pe #(
    parameter WIDTH = 16,       // Q8.8 Fixed Point
    parameter DT_SHIFT = 3      // dt = 2^-3 = 0.125 (Hardware optimization for division)
)(
    input  logic             clk,
    input  logic             rst_n,
    input  logic             enable,
    
    // Global Bifurcation Parameter a(t)
    input  logic signed [WIDTH-1:0] a_t,
    
    // Neighbor Coupling Input (Sum of J*x from neighbors)
    input  logic signed [WIDTH-1:0] coupling_force,
    
    // Output State (Spins)
    output logic signed [WIDTH-1:0] x_out,
    output logic                    sign_bit // The decoded bit (0 or 1)
);

    // --- Constants (Q8.8 Representation) ---
    // 1.0 = 256 (0x0100)
    // 0.7 = 179 (0x00B3) approx for Xi
    localparam signed [WIDTH-1:0] XI_CONST = 16'h00B3; 
    localparam signed [WIDTH-1:0] ONE      = 16'h0100;
    localparam signed [WIDTH-1:0] NEG_ONE  = 16'hFF00;

    // --- State Registers ---
    logic signed [WIDTH-1:0] x_reg, y_reg;
    logic signed [WIDTH-1:0] x_next, y_next;

    // --- Combinational Logic (The Critical Path) ---
    logic signed [WIDTH-1:0] delta_x;
    logic signed [WIDTH-1:0] delta_y;
    logic signed [WIDTH-1:0] force_term_1;
    logic signed [WIDTH-1:0] force_term_2;
    logic signed [WIDTH-1:0] total_force;
    logic signed [WIDTH-1:0] a_minus_xi;

    always_comb begin
        // 1. Update Position: x_next = x + (y * dt)
        // In hardware, multiply by dt is just a right shift if dt is power of 2
        delta_x = y_reg >>> DT_SHIFT; 
        x_next  = x_reg + delta_x;

        // 2. Wall Enforcement (The "Ballistic" part)
        // If |x| > 1.0, clamp it.
        // This is a simple mux in hardware.
        if (x_next > ONE)       x_next = ONE;
        else if (x_next < NEG_ONE) x_next = NEG_ONE;

        // 3. Calculate Force: F = -(a_t - xi)*x + xi*coupling
        a_minus_xi = a_t - XI_CONST;
        
        // Multiplications (These map to DSP48 Slices)
        // Result is Q16.16, we shift back to Q8.8
        force_term_1 = -((a_minus_xi * x_next) >>> 8); 
        force_term_2 = (coupling_force * XI_CONST) >>> 8;
        
        total_force = force_term_1 + force_term_2;

        // 4. Update Momentum: y_next = y + (force * dt)
        delta_y = total_force >>> DT_SHIFT;
        
        // Wall Bounce Logic (Simplified for RTL: Reset momentum if hitting wall)
        if (x_next == ONE || x_next == NEG_ONE) 
            y_next = 0;
        else 
            y_next = y_reg + delta_y;
    end

    // --- Sequential Logic ---
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_reg <= 0;
            y_reg <= 0;
        end else if (enable) begin
            x_reg <= x_next;
            y_reg <= y_next;
        end
    end

    // --- Output Assignments ---
    assign x_out = x_reg;
    assign sign_bit = ~x_reg[WIDTH-1]; // MSB indicates sign in 2's complement

endmodule
