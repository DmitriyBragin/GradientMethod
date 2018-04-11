#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <conio.h>
#include "MatrixWork.h"
#include <iostream>

#define EPSILON 0.001

//////////////////////////////////////////////////////////////
// MATH FUNCTION DECLARATION
//////////////////////////////////////////////////////////////

double Function(double x1, double x2)
{
	return 3.0 * x1 + x2 + 4.0 * sqrt(1.0 + x1 * x1 + 3.0 * x2 * x2);
}

void Gradient(double x[], double y[])
{
	double sqrt_1 = sqrt(1.0 + x[0] * x[0] + 3.0 * x[1] * x[1]);

	y[0] = 3.0 + (4.0 * x[0]) / sqrt_1;
	y[1] = 1.0 + (12.0 * x[1]) / sqrt_1;
}

void Hessian(double x[], double H[2][2])
{
	double sqr_1 = 1.0 + x[0] * x[0] + 3.0 * x[1] * x[1];
	double sqrt_1 = sqrt(sqr_1);

	H[0][0] = 4.0 * (1.0 - x[0] * x[0] / sqr_1) / sqrt_1;
	H[0][1] = H[1][0] = -12.0 * x[0] * x[1] / (sqr_1 * sqrt_1);
	H[1][1] = 12.0 * (1.0 - 3.0 * x[1] * x[1] / sqr_1) / sqrt_1;
}


void Gradient2(double x[], double y[], int recalculate)
{
	double **H, **invH, grad[2], det, tmp[2][2];
	H = new double*[2];
	invH = new double*[2];

	for (int i = 0; i < 2; i++)
	{
		H[i] = new double[2];
		invH[i] = new double[2];
	}

	if (recalculate)
	{
		Hessian(x, tmp);
		for (int i = 0; i < 2; i++)
		{
			for (int j = 0; j < 2; j++)
			{
				H[i][j] = tmp[i][j];
			}
		}
		inverse(H, invH, 2);
	}
	/* Умножение матриц */
	tmp[0][0] = H[0][0] * invH[0][0] + H[0][1] * invH[1][0];
	tmp[0][1] = H[0][0] * invH[0][1] + H[0][1] * invH[1][1];
	tmp[1][0] = H[1][0] * invH[0][0] + H[1][1] * invH[1][0];
	tmp[1][1] = H[1][0] * invH[0][1] + H[1][1] * invH[1][1];
	Gradient(x, grad);
	y[0] = (invH[0][0] * grad[0] + invH[0][1] * grad[1]);
	y[1] = (invH[1][0] * grad[0] + invH[1][1] * grad[1]);
}

//////////////////////////////////////////////////////////////
// OPTIMIZATIONS METHODS
//////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////
// STEP SUBDIVISION
//////////////////////////////////////////////////////////////
double Step_subdivide(double x[], double y[], double *f, double norm)
{
	double step = 2, fnew;

	if (norm < EPSILON)
		return 0;

	do
	{
		step *= 0.5;
		fnew = Function(x[0] - step * y[0], x[1] - step * y[1]);
	} while (fnew - *f > -EPSILON * step * norm * norm);

	x[0] -= step * y[0];
	x[1] -= step * y[1];
	*f = fnew;
	return step;
}

//////////////////////////////////////////////////////////////
// STEP NEWTON
//////////////////////////////////////////////////////////////

double Step_Newton(double x[], double y[], double *f, double norm)
{
	/* Pshenichniy power */
	double newtonStep = 1;
	double x_k_1[2];
	x_k_1[0] = x[0] - newtonStep * y[0];
	x_k_1[1] = x[1] - newtonStep * y[1];
	double fk1 = Function(x_k_1[0], x_k_1[1]);
	double fk2 = Function(x[0], x[1]);
	double mult;
	double grad[2];
	Gradient(x, grad);
	mult = -0.5 * (grad[0] * y[0] + grad[1] * y[1]);
	while ((fk1 - fk2) > (mult * newtonStep))
	{
		newtonStep /= 2;
		x_k_1[0] = x[0] - newtonStep * y[0];
		x_k_1[1] = x[1] - newtonStep * y[1];
		fk1 = Function(x_k_1[0], x_k_1[1]);
		fk2 = Function(x[0], x[1]);
		Gradient(x, grad);
		mult = -0.5 * (grad[0] * y[0] + grad[1] * y[1]);
	}

	x[0] -= newtonStep * y[0];
	x[1] -= newtonStep * y[1];
	
	*f = Function(x[0], x[1]);
	return newtonStep;
}

//////////////////////////////////////////////////////////////
// STEP FIXED LENGTH
//////////////////////////////////////////////////////////////
double lipsh = 12.0;
double m = 1e12, M = -1e12;
double Step_fixed(double x[], double y[], double *f, double norm)
{
	double step = (1 - EPSILON) / lipsh;
	//step = 2.0 / (m + M);
	m = 1.2;
	M = 12.0;
	step = 2.0 / (m + M);
	double **H, **invH, tmp[2][2];
	H = new double*[2];
	invH = new double*[2];

	for (int i = 0; i < 2; i++)
	{
		H[i] = new double[2];
		invH[i] = new double[2];
	}
	double t[2];
	t[0] = -1.1619;
	t[1] = -0.1291;
	Hessian(t, tmp);
	//step = 0.5;
	//step = 0.22;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			H[i][j] = tmp[i][j];
		}
	}
	inverse(H, invH, 2);
	double det = determinant(H, 2, 2);
	double h = invH[0][0];
	double u[2];
	u[0] = .89 * y[0] + .05 * y[1];
	u[1] = .05 * y[0] + .89 * y[1];
	x[0] -= step * u[0];
	x[1] -= step * u[1];
	*f = Function(x[0], x[1]);

	return step;
}

//////////////////////////////////////////////////////////////
// STEP OPTIMAL
//////////////////////////////////////////////////////////////
double Step_optimal(double x[], double y[], double *f, double norm)
{
	double alpha = (sqrt(5) - 1) / 2;
	double a = 0, b = 1;
	double l, m, fl, fm;

	l = a + (1 - alpha) * (b - a);
	m = a + b - l;
	fl = Function(x[0] - y[0] * l, x[1] - y[1] * l);
	fm = Function(x[0] - y[0] * m, x[1] - y[1] * m);

	while (fabs(m - l) > .000000001)
	{
		if (fl < fm)
		{
			b = m;
			m = l;
			fm = fl;
			l = a + (1 - alpha) * (b - a);
			fl = Function(x[0] - y[0] * l, x[1] - y[1] * l);
		}
		else
		{
			a = l;
			l = m;
			fl = fm;
			m = a + alpha * (b - a);
			fm = Function(x[0] - y[0] * m, x[1] - y[1] * m);
		}
	}
	double orta1[2], orta2[2];
	Gradient(x, orta1);
	x[0] -= l * y[0];
	x[1] -= l * y[1];
	Gradient(x, orta2);
	double res = orta1[0] * orta2[0] + orta1[1] * orta2[1];
	*f = Function(x[0], x[1]);
	return l;
}

//////////////////////////////////////////////////////////////
// STEP FLETCHER-REEVES
//////////////////////////////////////////////////////////////
int    fp_loc_iter;
double Step_FletcherReeves(double x[], double y[], double *f, double norm)
{
	double p_cur[2];
	double a = 0, step = .5;

	static double grad_prev[2] = { 0.0, 0.0 }, p_prev[2] = { 0.0, 0.0 };

	if (fp_loc_iter % 2 == 0)
	{
		p_cur[0] = -y[0];
		p_cur[1] = -y[1];
	}
	else
	{
		double betta;
		betta = (y[0] * y[0] + y[1] * y[1]) / (grad_prev[0] * grad_prev[0] + grad_prev[1] * grad_prev[1]);
		p_cur[0] = -y[0] + betta * p_prev[0];
		p_cur[1] = -y[1] + betta * p_prev[1];
	}
	grad_prev[0] = y[0];
	grad_prev[1] = y[1];
	p_prev[0] = p_cur[0];
	p_prev[1] = p_cur[1];
	fp_loc_iter++;

	while (step > 0.000001)
	{
		double f_prev, f_cur;
		int    i = 2;

		f_prev = Function(x[0] + (a + step) * p_cur[0], x[1] + (a + step) * p_cur[1]);
		while (i < 100)
		{
			f_cur = Function(x[0] + (a + step * i) * p_cur[0], x[1] + (a + step * i) * p_cur[1]);
			if (f_cur > f_prev) {
				break;
			}
			f_prev = f_cur; 
			i++;
		}
		a += step * (i - 2);
		step *= 0.1;
	}
	a += step;

	x[0] += a * p_cur[0];
	x[1] += a * p_cur[1];
	*f = Function(x[0], x[1]);

	return a;
}

//////////////////////////////////////////////////////////////
// TEST METHODS ROUTINE
//////////////////////////////////////////////////////////////
int period;
int Test(double(*Step)(double[], double[], double *, double), int i)
{
	int    step = 0;
	double x[2] = { -1, -1 }, x_prev[2], y[2],
		f = Function(x[0], x[1]), norm;
	double solution[2];
	double cond = 0;
	double condVec[2];
	do
	{
		Gradient(x, y);
		norm = sqrt(y[0] * y[0] + y[1] * y[1]);
		printf("x=(%06.4lf,%06.4lf) y=(%06.4lf, %06.4lf) norm=%08.6lf,"
			" f=%07.5lf", x[0], x[1], y[0], y[1], norm, f);
		if (norm <= EPSILON)
			break;
		if (i == 3)
			Gradient2(x, y, 1);

		memcpy(x_prev, x, 2 * sizeof(double));
		printf(" step=%6.5lf\n", Step(x, y, &f, norm));
		condVec[0] = x[0] - x_prev[0];
		condVec[1] = x[1] - x_prev[1];
		cond = sqrt(condVec[0] * condVec[0] + condVec[1] * condVec[1]);
		step++;
	} while (norm > EPSILON);

	if (i == 0)
		memcpy(solution, x, 2 * sizeof(double));
	printf("\nDone in %d steps, cond = %lf\n", step, cond);
//	getch();
	return step;
}

//////////////////////////////////////////////////////////////
// FIND LIPSHITZ CONSTANT
//////////////////////////////////////////////////////////////
double FindLipshitzConstant(double left, double right, double bottom, double top)
{
	double R, R_max = 0, x1[2], x2[2], y1[2], y2[2];
	double t, dx, dy, df, denom;
	int k = 0;
	double Rd = RAND_MAX;
	while (1)
	{
		t = (double)(rand() % 100) / 100.0;
		x1[0] = left + (1 - t) + right * t;
		t = (double)(rand() % 100) / 100.0;
		x1[1] = left + (1 - t) + right * t;
		t = (double)(rand() % 100) / 100.0;
		x2[0] = bottom + (1 - t) + top * t;
		t = (double)(rand() % 100) / 100.0;
		x2[1] = bottom + (1 - t) + top * t;

		dx = x2[0] - x1[0];
		dy = x2[1] - x1[1];
		Gradient(x1, y1);
		Gradient(x2, y2);
		df = sqrt((y2[0] - y1[0]) * (y2[0] - y1[0]) + (y2[1] - y1[1]) * (y2[1] - y1[1]));

		denom = sqrt(dx * dx + dy * dy);
		if (denom < EPSILON)
			continue;
		R = df / denom;
		k++;
		if (R > R_max)
		{
			R_max = R;
			printf("%lf\n", R_max);
		}
	}
}

//////////////////////////////////////////////////////////////
// FIND EIGEN VALUES (m and M)
//////////////////////////////////////////////////////////////
void FindEigenVals(double left, double right, double bottom, double top)
{
	int    i, j, num = 100;
	double x[2], H[2][2];
	double p, q, descr, x1, x2;

	for (i = 0; i <= num; i++)
	{
		x[0] = left + ((right - left) * i) / num;
		for (j = 0; j <= num; j++)
		{
			x[1] = bottom + ((top - bottom) * j) / num;
			Hessian(x, H);
			p = -H[0][0] - H[1][1];
			q = H[0][0] * H[1][1] - H[0][1] * H[1][0];
			descr = p * p - 4 * q;
			if (descr < 0)
			{
				printf("SHIT\n");
				continue;
			}
			descr = sqrt(descr);
			x1 = -0.5 * p - 0.5 * descr;
			x2 = x1 + descr;
			if (x1 < m)
			{
				m = x1;
			}
			if (x2 > M)
			{
				M = x2;
			}
		}
	}
	printf("m=%08.6lf; M=%08.6lf\n", m, M);
	//getch();
}

//////////////////////////////////////////////////////////////
// MAIN FUNCTION
//////////////////////////////////////////////////////////////
int main(void)
{
	/*
	* NOTE: IF YOU WANT TO USE THIS CODE, CHANGE THOSE THINGS:
	* FOR SQARE FORM MINIMIZATION AND NEWTON
	* 1) Function() <- this is for standart square form (x^2 + y^2 + z^2 + xy + xz + yz), change your coefficents manually
	* 2) Gradient() <- change this for your gradient of square form
	* 3) Hessian() <- change for your Hessian matrix
	*/
	int i;
//	FindEigenVals(-2, 2, -2, 2);
	//FindLipshitzConstant(-2, 2, -2, 2);
	printf("\n2-dimensional minimization example\n");
	printf("\nFixed step:\n");
	Test(Step_fixed, 0);
	printf("\nOptimal step:\n");
	Test(Step_optimal, 2);
	printf("\nSecond order gradient:\n");
	Test(Step_Newton, 3);

	return 0;
}
