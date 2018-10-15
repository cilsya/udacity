#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// https://stackoverflow.com/questions/1727881/how-to-use-the-pi-constant-in-c
//const double PI = 3.14159265358979323846;
const double PI = M_PI;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
  
  // From Lesson 13 - Laser Measurements Part 3
  // kalman_filter.cpp
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  
  // From Lesson 13 - Laser Measurements Part 3
  // kalman_filter.cpp
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
  
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
  
  // Lesson 15 - Radar Measurements
  // Range: rho
  // Bearing: phi
  // Radial Velocity = rho_dot
  //
  // Equations are also in this same lesson
  float rho = sqrt( pow(x_(0),2) + pow(x_(1), 2) );
  float phi = atan2(x_(1), x_(0));
  float rho_dot = (x_(0)*x_(2) + x_(1)*x_(3)  )/rho;

  // From Lesson 7 - Tips and Tricks for this project
  // Avoid divide by zero.
  //float tolerance = 0.000001;
  //
  // If rho is below tolerance,
  // the value for tolerance will become rho's value
  //rho = std::max(rho, tolerance);
  if (rho <= 0.000001){
    rho = 0.000001;
  }
  
  VectorXd z_pred =  VectorXd(3);
  z_pred << rho, phi, rho_dot;
  
  VectorXd y = z - z_pred;
  //
  // From Lesson 7 - Tips and Tricks for this project
  // Normalizing Angles.
  // NOTE: Can probably use modulo but using while loops
  //       instead of dealing with radians
  while( y(1) > PI )
  {
    y(1) -= 2*PI; 
  }
  
  while( y(1) < -PI )
  {
    y(1) += 2*PI; 
  }
  
  // Based on Lesson 13 - Laser Measurements Part 3
  // kalman_filter.cpp
  //VectorXd z_pred = H_ * x_;
  //VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
