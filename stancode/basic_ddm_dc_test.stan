
functions { 
  /* Wiener diffusion log-PDF for a single response (adapted from brms 1.10.2)
   * Arguments: 
   *   Y: acc*rt in seconds (negative and positive RTs for incorrect and correct responses respectively)
   *   boundary: boundary separation parameter > 0
   *   ter: non-decision time parameter > 0
   *   bias: initial bias parameter in [0, 1]
   *   drift: drift rate parameter
   *   dc: diffusion coefficient parameter
   * Returns:  
   *   a scalar to be added to the log posterior 
   */ 
   real diffusion_lpdf(real Y, real boundary, 
                              real ter, real bias, real drift, real dc) { 
    
    if (fabs(Y) < ter) {
        return wiener_lpdf( ter+0.0001 | boundary/dc, ter, bias, drift/dc ); // does this work?
    } else {
        if (Y >= 0) {
            return wiener_lpdf( fabs(Y) | boundary/dc, ter, bias, drift/dc );
        } else {
            return wiener_lpdf( fabs(Y) | boundary/dc, ter, 1-bias, -drift/dc );
        }
    }

   }
} 
data {
    int<lower=1> N; // Number of trial-level observations
    int<lower=1> nparts; // Number of participants
    real y[N]; // acc*rt in seconds (negative and positive RTs for incorrect and correct responses respectively)
    int<lower=1> participant[N]; // Participant index
}
parameters {
    vector<lower=0, upper=10>[nparts] alpha; // Boundary parameter (speed-accuracy tradeoff)
    vector<lower=0, upper=1.5>[nparts] ndt; // Non-decision time
    vector<lower=0, upper=1>[nparts] beta; // Start point bias towards choice A
    vector[nparts] delta; // Drift rate to choice A
    vector<lower=0, upper=10>[nparts] varsigma; // Diffusion coefficient per participant
}
model {

    // ##########
    // Participant-level DDM parameter priors
    // ##########
    for (p in 1:nparts) {

        // Boundary parameter (speed-accuracy tradeoff) per participant
        alpha[p] ~ normal(1, .5) T[0, 10];

        // Non-decision time per participant
        ndt[p] ~ normal(.5, .25) T[0, 1.5];

        // Start point bias towards choice A per participant
        beta[p] ~ beta(2, 2);

        // Drift rate to choice A per participant
        delta[p] ~ normal(0, 2);

        // Diffusion coefficient per participant
        varsigma[p] ~ normal(1, .5) T[0, 10];

    }
    // Wiener likelihood
    for (i in 1:N) {

        target += diffusion_lpdf( y[i] | alpha[participant[i]], 
            ndt[participant[i]], beta[participant[i]], delta[participant[i]], varsigma[participant[i]]);
    }
}
