

void A(double localTensor[3][3], double dt, double detwei[6], double CG1[3][6])
{
  for(int i_r_0 = 0; i_r_0 < 3; i_r_0++)
  {
    for(int i_r_1 = 0; i_r_1 < 3; i_r_1++)
    {
      localTensor[i_r_0][i_r_1] = 0.0;
      for(int i_g = 0; i_g < 6; i_g++)
      {
        localTensor[i_r_0][i_r_1] += CG1[i_r_0][i_g] * CG1[i_r_1][i_g] * detwei[i_g];
      };
    };
  };
}

void RHS(double localTensor[3], double dt, double detwei[6], double c0[3], double CG1[3][6])
{
  for(int i_r_0 = 0; i_r_0 < 3; i_r_0++)
  {
    double c_q0[6];
    for(int i_g = 0; i_g < 6; i_g++)
    {
      c_q0[i_g] = 0.0;
      for(int i_r_0 = 0; i_r_0 < 3; i_r_0++)
      {
        c_q0[i_g] += c0[i_r_0] * CG1[i_r_0][i_g];
      };
    };
    localTensor[i_r_0] = 0.0;
    for(int i_g = 0; i_g < 6; i_g++)
    {
      localTensor[i_r_0] += CG1[i_r_0][i_g] * c_q0[i_g] * detwei[i_g];
    };
  };
}



