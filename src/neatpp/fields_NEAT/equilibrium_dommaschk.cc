// ::gyronimo:: - gyromotion for the people, by the people -
// An object-oriented library for gyromotion applications in plasma physics.
// Copyright (C) 2021 Paulo Rodrigues.

// ::gyronimo:: is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// ::gyronimo:: is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with ::gyronimo::.  If not, see <https://www.gnu.org/licenses/>.

#include "equilibrium_dommaschk.hh"
#include <gyronimo/core/dblock.hh>
#include <math.h>

double alpha(int m,int l) {
	double y;
    if (l < 0) {
		y=0;
	}
	else {
		y = pow(-1.0,l)/(tgamma(m+l+1)*tgamma(l+1)*pow(2.0,2*l+m));
    }
	return y;
}

double alphas(int m,int l) {
	double y;
	y = (2*l+m)*alpha(m,l);
	return y;
}

double beta(int m,int l) {
	double y;
    if (l < 0 || l >=m) {
		y=0;
	}
	else {
		y = tgamma(m-l)/(tgamma(l+1)*pow(2.0,2*l-m+1));
    }
	return y;
}

double betas(int m,int l) {
	double y;
	y = (2*l-m)*beta(m,l);
	return y;
}

double gamma1(int m,int l) {
	double y;
    if (l <= 0) {
		y=0;
	}
	else {
        double sumN = 0.0;
        for(int i = 1; i <= l; ++i)
        {
            sumN += 1.0/(i) + 1.0/(m+i);
        }
		y = (alpha(m,l)/2)*sumN;
    }
	return y;
}

double gammas(int m,int l) {
	double y;
	y = (2*l+m)*gamma1(m,l);
	return y;
}

double Dmn(int m, int n, double R, double Z) {
	double sumD=0.0, y=0.0;
	int j, k;
	for (k=0; k<=floor(n/2); k++) {
		sumD = 0.0;
		for (j=0;j<k+1;j++) {
			sumD += -(alpha(m,j)*(alphas(m,k-m-j)*log(R)+gammas(m,k-m-j)-alpha(m,k-m-j))-gamma1(m,j)*alphas(m,k-m-j)+alpha(m,j)*betas(m,k-j))*pow(R,(2*j+m)) + alphas(m,k-j)*beta(m,j)*pow(R,2*j-m);
		}
		y += (pow(Z,n-2*k)/tgamma(n-2*k+1))*sumD;
	}
	return y;
}

double Nmn(int m, int n, double R, double Z) {
	double sumN = 0.0, y = 0.0;
	int j, k;
	for (k=0;k<=floor(n/2); k++) {
		sumN = 0.0;
		for (j=0;j<k+1;j++) {
            sumN += +(alpha(m,j)*(alpha(m,k-m-j)*log(R)+gamma1(m,k-m-j))-gamma1(m,j)*alpha(m,k-m-j)+alpha(m,j)*beta(m,k-j))*pow(R,(2*j+m)) - alpha(m,k-j)*beta(m,j)*pow(R,2*j-m);
		}
		y += (pow(Z,n-2*k)/tgamma(n-2*k+1))*sumN;
	}
	return y;
}

double dRDmn(int m, int n, double R, double Z) {
	double sumD= 0.0, y=0.0;
	int j, k;
	for (k=0; k<=floor(n/2); k++) {
		sumD = 0.0;
		for (j=0;j<k+1;j++) {
		//	sumD += -(alpha(m,j)*(alphas(m,k-m-j)*log(R)+gammas(m,k-m-j)-alpha(m,k-m-j)+alphas(m,k-m-j)/(2*j+m))-gamma1(m,j)*alphas(m,k-m-j)+alpha(m,j)*betas(m,k-j))*pow(R,(2*j+m-1))*(2*j+m) + alphas(m,k-j)*beta(m,j)*pow(R,2*j-m-1)*(2*j-m);
			sumD += -(alpha(m,j)*(alphas(m,k-m-j)*(1.0/(2*j+m)+log(R))+gammas(m,k-m-j)-alpha(m,k-m-j))-gamma1(m,j)*alphas(m,k-m-j)+alpha(m,j)*betas(m,k-j))*pow(R,(2*j+m-1))*(2*j+m) + alphas(m,k-j)*beta(m,j)*pow(R,2*j-m-1)*(2*j-m);
		}
		y += (pow(Z,n-2*k)/tgamma(n-2*k+1))*sumD;
	}
	return y;
}

double dZDmn(int m, int n, double R, double Z) {
	double sumD= 0.0, y=0.0;
	int j, k;
	for (k=0;k<=floor(n/2); k++) {
		sumD = 0.0;
		for (j=0; j<k+1;j++) {
			sumD += -(alpha(m,j)*(alphas(m,k-m-j)*log(R)+gammas(m,k-m-j)-alpha(m,k-m-j))-gamma1(m,j)*alphas(m,k-m-j)+alpha(m,j)*betas(m,k-j))*pow(R,(2*j+m)) + alphas(m,k-j)*beta(m,j)*pow(R,2*j-m);
		}
        if (n-2*k==0){
            y += 0;
        }
        else {
		    y += ((n-2*k)*pow(Z,n-2*k-1)/tgamma(n-2*k+1))*sumD;
        }
	}
	return y;
}

double dRRDmn(int m, int n, double R, double Z) {
	double sumD= 0.0, y=0.0;
	int j, k;
	for (k=0;k<=floor(n/2);k++) {
		sumD = 0.0;
		for (j=0; j<k+1; j++) {
			sumD += -(alpha(m,j)*(alphas(m,k-m-j)*(((4.0*j+2*m-1)/((2.0*j+m)*(2.0*j+m-1)))+log(R))+gammas(m,k-m-j)-alpha(m,k-m-j))-gamma1(m,j)*alphas(m,k-m-j)+alpha(m,j)*betas(m,k-j))*pow(R,(2*j+m-2))*(2*j+m)*(2*j+m-1) + alphas(m,k-j)*beta(m,j)*pow(R,2*j-m-2)*(2*j-m)*(2*j-m-1);
		}
		y += (pow(Z,n-2*k)/tgamma(n-2*k+1))*sumD;
	}
	return y;
}

double dZZDmn(int m, int n, double R, double Z) {
	double sumD= 0.0, y=0.0;
	int j, k;
	for (k=0;k<=floor(n/2);k++) {
		sumD = 0.0;
		for (j=0; j<k+1; j++) {
			sumD += -(alpha(m,j)*(alphas(m,k-m-j)*log(R)+gammas(m,k-m-j)-alpha(m,k-m-j))-gamma1(m,j)*alphas(m,k-m-j)+alpha(m,j)*betas(m,k-j))*pow(R,(2*j+m)) + alphas(m,k-j)*beta(m,j)*pow(R,2*j-m);
		}
        if (n-2*k==0 || n-2*k-1==0){
            y += 0;
        }
        else {
		    y += ((n-2*k-1)*pow(Z,n-2*k-2)/tgamma(n-2*k))*sumD;
        }
	}
	return y;
}

double dRZDmn(int m, int n, double R, double Z) {
	double sumD= 0.0, y=0.0;
	int j, k;
	for (k=0;k<=floor(n/2);k++) {
		sumD = 0.0;
		for (j=0; j<k+1; j++) {
			sumD += -(alpha(m,j)*(alphas(m,k-m-j)*(1.0/(2*j+m)+log(R))+gammas(m,k-m-j)-alpha(m,k-m-j))-gamma1(m,j)*alphas(m,k-m-j)+alpha(m,j)*betas(m,k-j))*pow(R,(2*j+m-1))*(2*j+m) + alphas(m,k-j)*beta(m,j)*pow(R,2*j-m-1)*(2*j-m);
		}
        if (n-2*k==0){
            y += 0;
        }
        else {
		    y += (pow(Z,n-2*k-1)/tgamma(n-2*k))*sumD;
        }
	}
	return y;
}

double dRNmn(int m, int n, double R, double Z) {
	double sumN= 0.0, y=0.0;
	int j, k;
	for (k=0;k<=floor(n/2); k++) {
		sumN = 0.0;
		for (j=0; j<k+1; j++) {
		//	sumN += +(alpha(m,j)*(alpha(m,k-m-j)*log(R)+gamma1(m,k-m-j)+alpha(m,k-m-j)/(2*j+m))-gamma1(m,j)*alpha(m,k-m-j)+alpha(m,j)*beta(m,k-j))*pow(R,(2*j+m-1))*(2*j+m) - alpha(m,k-j)*beta(m,j)*pow(R,2*j-m-1)*(2*j-m);
			sumN += +(alpha(m,j)*(alpha(m,k-m-j)*(1.0/(2*j+m)+log(R))+gamma1(m,k-m-j))-gamma1(m,j)*alpha(m,k-m-j)+alpha(m,j)*beta(m,k-j))*pow(R,(2*j+m-1))*(2*j+m) - alpha(m,k-j)*beta(m,j)*pow(R,2*j-m-1)*(2*j-m);

		}	
	y += (pow(Z,n-2*k)/tgamma(n-2*k+1))*sumN;
	}
	return y;
}

double dZNmn(int m, int n, double R, double Z) {
	double sumN= 0.0, y=0.0;
	int j, k;
	for (k=0;k<=floor(n/2);k++) {
		sumN = 0.0;
		for (j=0; j<k+1; j++) {
			sumN += +(alpha(m,j)*(alpha(m,k-m-j)*log(R)+gamma1(m,k-m-j))-gamma1(m,j)*alpha(m,k-m-j)+alpha(m,j)*beta(m,k-j))*pow(R,(2*j+m)) - alpha(m,k-j)*beta(m,j)*pow(R,2*j-m);
		}
        if (n-2*k==0 || n<0) {
            y += 0;
        }
        else {
		    y += ((n-2*k)*pow(Z,n-2*k-1)/tgamma(n-2*k+1))*sumN;
        }
	}
	return y;
}

double dRRNmn(int m, int n, double R, double Z) {
	double sumN= 0.0, y=0.0;
	int j, k;
	for (k=0; k< n/2 + 1; k++) {
		sumN = 0.0;
		for (j=0; j< k+1; j++) {
			sumN += +(alpha(m,j)*(alpha(m,k-m-j)*(((4.0*j+2*m-1)/((2.0*j+m)*(2.0*j+m-1)))+log(R))+gamma1(m,k-m-j))-gamma1(m,j)*alpha(m,k-m-j)+alpha(m,j)*beta(m,k-j))*pow(R,(2*j+m-2))*(2*j+m)*(2*j+m-1) - alpha(m,k-j)*beta(m,j)*pow(R,2*j-m-2)*(2*j-m)*(2*j-m-1);
		}
		y += (pow(Z,n-2*k)/tgamma(n-2*k+1))*sumN;
	}
	return y;
}

double dZZNmn(int m, int n, double R, double Z) {
	double sumN= 0.0, y=0.0;
	int j, k;
	for (k=0; k< n/2; k++) {
		sumN = 0.0;
		for (j =0; j<k+1; j++) {
			sumN += +(alpha(m,j)*(alpha(m,k-m-j)*log(R)+gamma1(m,k-m-j))-gamma1(m,j)*alpha(m,k-m-j)+alpha(m,j)*beta(m,k-j))*pow(R,(2*j+m)) - alpha(m,k-j)*beta(m,j)*pow(R,2*j-m);
		}
        if (n-2*k==0 || n-2*k-1==0 || n<0) {
            y += 0;
        }
        else {
		    y += ((n-2*k-1)*pow(Z,n-2*k-2)/tgamma(n-2*k))*sumN;
        }
	}
	return y;
}

double dRZNmn(int m, int n, double R, double Z) {
	double sumN= 0.0, y=0.0;
	int j, k;
	for (k=0;k<=floor(n/2); k++) {
		sumN = 0.0;
		for (j=0; j< k+1; j++) {
			sumN += +(alpha(m,j)*(alpha(m,k-m-j)*(1.0/(2*j+m)+log(R))+gamma1(m,k-m-j))-gamma1(m,j)*alpha(m,k-m-j)+alpha(m,j)*beta(m,k-j))*pow(R,(2*j+m-1))*(2*j+m) - alpha(m,k-j)*beta(m,j)*pow(R,2*j-m-1)*(2*j-m);
		}
        if (n-2*k==0 || n<0) {
            y += 0;
        }
        else {
		    y += ((n-2*k)*pow(Z,n-2*k-1)/tgamma(n-2*k+1))*sumN;
        }
	}
	return y;
}

double Phi(int m, int nn, double R, double Z, double phi,double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (nn%2 == 0) {
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1; 
		d = coeff2; 
		b = c = 0;
	}
	y = (a*cos(m*phi) + b*sin(m*phi))*Dmn(m,nn,R,Z) + (c*cos(m*phi) + d*sin(m*phi))*Nmn(m,nn-1,R,Z);
	return y;
}

double BR(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0) {
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1;
		d = coeff2; 
		b = c = 0;
	}
	y = (a*cos(m*phi) + b*sin(m*phi))*dRDmn(m,n,R,Z) + (c*cos(m*phi) + d*sin(m*phi))*dRNmn(m,n-1,R,Z);
	return y;
}

double BZ(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0) {
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1;
		d = coeff2; 
		b = c = 0;
	}
	y = (a*cos(m*phi) + b*sin(m*phi))*dZDmn(m,n,R,Z) + (c*cos(m*phi) + d*sin(m*phi))*dZNmn(m,n-1,R,Z);
	return y;
}

double Bphi(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0) {
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1;
		d = coeff2; 
		b = c = 0;
	}
	y = m*(-a*sin(m*phi) + b*cos(m*phi))*Dmn(m,n,R,Z)/R + m*(-c*sin(m*phi) + d*cos(m*phi))*Nmn(m,n-1,R,Z)/R;
	return y;
}

double dphiBR(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0) {
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1;
		d = coeff2; 
		b = c = 0;
	}
	y = m*((-a*sin(m*phi) + b*cos(m*phi))*dRDmn(m,n,R,Z) + (-c*sin(m*phi) + d*cos(m*phi))*dRNmn(m,n-1,R,Z));
	return y;
}

double dphiBZ(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0) {
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1;
		d = coeff2; 
		b = c = 0;
	}
	y = m*((-a*sin(m*phi) + b*cos(m*phi))*dZDmn(m,n,R,Z) + (-c*sin(m*phi) + d*cos(m*phi))*dZNmn(m,n-1,R,Z));
	return y;
}

double dphiBphi(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0) {
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1;
		d = coeff2; 
		b = c = 0;
	}
	y = m*(m*(-a*cos(m*phi) - b*sin(m*phi))*Dmn(m,n,R,Z)/R + m*(-c*cos(m*phi) - d*sin(m*phi))*Nmn(m,n-1,R,Z)/R);
	return y;
}

double dRBR(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0) {
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1;
		d = coeff2; 
		b = c = 0;
	}
	y = (a*cos(m*phi) + b*sin(m*phi))*dRRDmn(m,n,R,Z) + (c*cos(m*phi) + d*sin(m*phi))*dRRNmn(m,n-1,R,Z);
	return y;
}

double dZBZ(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0) {
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1;
		d = coeff2; 
		b = c = 0;
	}
	y = (a*cos(m*phi) + b*sin(m*phi))*dZZDmn(m,n,R,Z) + (c*cos(m*phi) + d*sin(m*phi))*dZZNmn(m,n-1,R,Z);
	return y;
}

double dRBZ(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0) {
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1;
		d = coeff2; 
		b = c = 0;
	}
	y = (a*cos(m*phi) + b*sin(m*phi))*dRZDmn(m,n,R,Z) + (c*cos(m*phi) + d*sin(m*phi))*dRZNmn(m,n-1,R,Z);
	return y;
}

double dZBR(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0) {
		a = d = 0; 
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1;
		d = coeff2; 
		b = c = 0;
	}
	y = (a*cos(m*phi) + b*sin(m*phi))*dRZDmn(m,n,R,Z) + (c*cos(m*phi) + d*sin(m*phi))*dRZNmn(m,n-1,R,Z);
	return y;
}

double dRBphi(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0) {
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1; 
		d = coeff2;
		b = c = 0;
	}
	y =  m*(-a*sin(m*phi) + b*cos(m*phi))*dRDmn(m,n,R,Z)/R + m*(-c*sin(m*phi) + d*cos(m*phi))*dRNmn(m,n-1,R,Z)/R - m*(-a*sin(m*phi) + b*cos(m*phi))*Dmn(m,n,R,Z)/pow(R,2) - m*(-c*sin(m*phi) + d*cos(m*phi))*Nmn(m,n-1,R,Z)/pow(R,2);
	return y;
}

double dZBphi(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0){
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1; 
		d = coeff2;
		b = c = 0;
	}
	y = m*(-a*sin(m*phi) + b*cos(m*phi))*dZDmn(m,n,R,Z)/R + m*(-c*sin(m*phi) + d*cos(m*phi))*dZNmn(m,n-1,R,Z)/R;
	return y;
}

namespace gyronimo{

equilibrium_dommaschk::equilibrium_dommaschk(
      const metric_cylindrical *g, int m, int l, double coeff1, double coeff2, double B0)
    : IR3field_c1(B0, 1.0, g),
      metric_(g), m_(m), l_(l), coeff1_(coeff1), coeff2_(coeff2), B0_(B0) {
}


IR3 equilibrium_dommaschk::contravariant(const IR3& position, double time) const {
  	double R = position[IR3::u];
  	double phi = position[IR3::v];
  	double Z = position[IR3::w];
	return {
		BR(m_, l_, R, Z, phi, coeff1_, coeff2_),
		Bphi(m_, l_, R, Z, phi, coeff1_, coeff2_)+1/(R),//o código do Bphi já divide por R uma vez, não sei se o devo fazer só mais uma vez ou não de todo
		BZ(m_, l_, R, Z, phi, coeff1_, coeff2_)	
	};
}

dIR3 equilibrium_dommaschk::del_contravariant(
    const IR3& position, double time) const {
  	double R = position[IR3::u];
  	double phi = position[IR3::v];
  	double Z = position[IR3::w];
  	return {
		dRBR(m_,l_,R,Z,phi,coeff1_,coeff2_), dphiBR(m_,l_,R,Z,phi,coeff1_,coeff2_), dZBR(m_,l_,R,Z,phi,coeff1_,coeff2_),
		dRBphi(m_,l_,R,Z,phi,coeff1_,coeff2_)-1/(R*R), dphiBphi(m_,l_,R,Z,phi,coeff1_,coeff2_), dZBphi(m_,l_,R,Z,phi,coeff1_,coeff2_),
		dRBZ(m_,l_,R,Z,phi,coeff1_,coeff2_), dphiBZ(m_,l_,R,Z,phi,coeff1_,coeff2_), dZBZ(m_,l_,R,Z,phi,coeff1_,coeff2_)
	};	 //verificar se esta matriz não está transposta em relação ao que é suposto

}

}// end namespace gyronimo.
