void integrate_parm(float *  *  u_em_3_1_out, float *  e_0_0, float *  e_1_0, float *  h_2_0, float *  u_em_3_0, float *  u_em_3_1, float MU, float EPSILON, int x_max, int y_max, int cb_x, int cb_y, int chunk, int _unroll_p3);
void integrate(float *  *  u_em_3_1_out, float *  e_0_0, float *  e_1_0, float *  h_2_0, float *  u_em_3_0, float *  u_em_3_1, float MU, float EPSILON, int x_max, int y_max);
void initialize_integrate_parm(float *  e_0_0, float *  e_1_0, float *  h_2_0, float *  u_em_3_0, float *  u_em_3_1, float MU, float EPSILON, int x_max, int y_max, int cb_x, int cb_y, int chunk);
void initialize_integrate(float *  e_0_0, float *  e_1_0, float *  h_2_0, float *  u_em_3_0, float *  u_em_3_1, float MU, float EPSILON, int x_max, int y_max);
