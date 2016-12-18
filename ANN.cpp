// ANN.cpp : ¶¨Òå¿ØÖÆÌ¨Ó¦ÓÃ³ÌÐòµÄÈë¿Úµã¡£
//
#include "stdafx.h"
#include <stdio.h>  
#include <stdlib.h>  
#include <math.h>  
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
using namespace std;

int sign(double x)
{
	if(x == 0) return 0;
	else if(x > 0) return 1;
	else return -1;
}
int data_nom_input(vector<vector<double> >& data)
{
	if(data.size() == 0) return -1;
	for(int i=0; i<data.size(); ++i)
	{
		double max_ele = *max_element(data[i].begin(), data[i].end());
		double min_ele = *min_element(data[i].begin(), data[i].end());
		for(int j=0; j<data[0].size(); ++j)
		{
			data[i][j] = (data[i][j] - min_ele) / (max_ele - min_ele);
		}
	}

	return 0;
}
int data_nom_output(vector<vector<double> >& data, bool isClassify)
{
	int kClassify = 0;
	if(isClassify)
	{
		for(int i=0; i<data.size(); ++i)
		{
			for(int j=0; j<data[0].size(); ++j)
			{
				if(data[i][j] > kClassify)
					kClassify = data[i][j];
			}
		}

		for(int i=0; i<data.size(); ++i)
		{
			for(int j=0; j<data[0].size(); ++j)
			{
				data[i][j] = 1.0/kClassify/2 + (data[i][j] - 1) * 1.0/kClassify;
			}
		}

	}
	else
	{
		for(int i=0; i<data.size(); ++i)
		{
			double max_ele = *max_element(data[i].begin(), data[i].end());
			double min_ele = *min_element(data[i].begin(), data[i].end());
			for(int j=0; j<data[0].size(); ++j)
			{
				data[i][j] = (data[i][j] - min_ele) / (max_ele - min_ele);
			}
		}
	}

	return kClassify;
}
int NN_data_preproces(const char* data, vector<vector<double> >& input, vector<vector<double> >& output_e, int input_dim, int output_dim)
{
	vector<double> every_input(input_dim);
	vector<double> every_output(output_dim);
	string line;
	ifstream fp;
	int temp_input, temp_output, position;

	fp.open(data, ios::in);

	if(!fp.is_open())
	{
		cout<<"The data file not open correctly"<<endl;
		return -1;
	}
	//int cnt = 0;
	while(getline(fp, line))
	{
		temp_input = 0;
		temp_output = 0;
		position = 0;

		for(int i=0; i<line.length(); ++i)
		{
			if(line[i] == ',')
			{
				if(temp_input < input_dim)
				{
					string tempstr = line.substr(position, i-position);
					char* tempchar = new char[tempstr.length()];

					for(int i=0; i<tempstr.length(); ++i)
						tempchar[i] = tempstr[i];

					every_input[temp_input++] = atof(tempchar);

					delete []tempchar;
				}
				else
				{
					string tempstr = line.substr(position, i-position);
					char* tempchar = new char[tempstr.length()];

					for(int i=0; i<tempstr.length(); ++i)
						tempchar[i] = tempstr[i];

					every_output[temp_output++] = atof(tempchar);

					delete []tempchar;
				}
				position = i+1;
			}
			else if(i == line.length()-1)
			{
					string tempstr = line.substr(position, i-position+1);
					char* tempchar = new char[tempstr.length()];

					for(int i=0; i<tempstr.length(); ++i)
						tempchar[i] = tempstr[i];

					every_output[temp_output++] = atof(tempchar);

					delete []tempchar;

			}

		}

		input.push_back(every_input);
		output_e.push_back(every_output);
		//cnt++;
	}

	fp.close();
	fp.clear();

	return 0;
}

void NN_train(const char* train_data, const char* test_data, const char* validation_data, int hide_num, int input_dim, int output_dim, double learnrate, bool isClassify_or_R, int maxtrainnum,double E_error, double allowed_error = 0)
{
	vector<vector<double> > input;    //train data
	vector<vector<double> > output_e; //expected output
	vector<vector<double> > v_input;  //validation data
	vector<vector<double> > v_output_e; //expected output
	double error = 0;
	double validation_error = 0;
	int notimprove_cnt = 0;
	double min_verror = 100000000;
	int trainnum = 0;
	int times_lessthanmin = 0;
	double minerror;
	
	
	NN_data_preproces(train_data, input, output_e, input_dim, output_dim);
	NN_data_preproces(validation_data, v_input, v_output_e, input_dim, output_dim);
	data_nom_input(input);
	data_nom_input(v_input);
	int kClassify = data_nom_output(output_e, isClassify_or_R);
	data_nom_output(v_output_e, isClassify_or_R);

	vector<double> weight((output_dim + input_dim) * hide_num, 0);
	vector<double> bias(output_dim + hide_num, 0);
	vector<double> hide_output(hide_num);
	vector<double> final_output(output_dim);
	int weight_output;
	int weight_hide;
	int weight_input;
	int weight_hide_to_output;
	int temp_weight;
	int bias_output;
	int rand_train_data = rand()%input.size(); //minibatch

	//for(int i=0; i<(output_dim + input_dim) * hide_num; ++i)
	//{
	//	weight[i] = rand();
	//}

	//for(int i=0; i<output_dim + hide_num; ++i)
	//{
	//	bias[i] = rand();
	//}

	for(int i=0; i<output_e.size(); ++i)
	{
		for(int j=0; j<output_dim; ++j)
			error += output_e[i][j] * output_e[i][j];
	}

	error /= input.size();

	minerror = error;

	double preerror = error;
	double temp_gradient;

	while(error > E_error && trainnum < maxtrainnum)
	{
	    weight_output = 0;
	    weight_hide = 0;
	    weight_input = 0;
	    weight_hide_to_output = 0;
 	    temp_weight = hide_num*output_dim;
    
	    for(int i=0; i<hide_num; ++i)
	    {
	    	for(int j=0; j<input_dim; ++j)
	    	{
	    		hide_output[i] += weight[temp_weight++] * input[rand_train_data][j];
	    	}

	    	hide_output[i] -= bias[i];
	    	hide_output[i] = 1.0*1/(1+exp(-hide_output[i]));
	    }

	    temp_weight = 0;

	    for(int i=0; i<output_dim; ++i)
	    {
	    	for(int j=0; j<hide_num; ++j)
	    	{
	    		final_output[i] += weight[temp_weight++] * hide_output[j];
	    	}

	    	final_output[i] -= bias[hide_num + i];
	    	final_output[i] = 1.0*1/(1+exp(-final_output[i]));
	    }

	    for(int i=0; i<output_dim+hide_num; ++i)
	    {
	    	bias_output = i;
	    	if(i < hide_num)
	    	{
	    		for(int j=0; j<output_dim; ++j)
	    		{
				temp_gradient = hide_output[i]*(1-hide_output[i])*(final_output[j] - output_e[rand_train_data][j])*weight[bias_output]*
						final_output[j]*(1-final_output[j]);
	    			bias[i] += learnrate * temp_gradient;
	    			bias_output += hide_num;
	    		}
	    	}
	    	else
	    	{
			temp_gradient = final_output[i-hide_num]*(1-final_output[i-hide_num])*
					(final_output[i-hide_num] - output_e[rand_train_data][i-hide_num]);
	    		bias[i] += learnrate * temp_gradient;
	    	}
	    } 

		for(int i=0; i<(output_dim+input_dim)*hide_num; ++i)
		{
			if(i < hide_num*output_dim)
			{
				temp_gradient = hide_output[weight_hide]*(final_output[weight_output]-output_e[rand_train_data][weight_output])*
					final_output[weight_output]*(1-final_output[weight_output]);
				weight[i] -= learnrate * temp_gradient;
				weight_hide++;
				weight_hide %= hide_num;
				if(weight_hide == 0)
						weight_output++;
			}
			else
			{
				weight_output = 0;
				weight_hide_to_output = 0;
				for(int j=0; j<output_dim; ++j)
				{
					temp_gradient = hide_output[weight_hide]*input[rand_train_data][weight_input]*(final_output[weight_output]-output_e[rand_train_data][weight_output])*
								 (1-hide_output[weight_hide])*weight[weight_hide_to_output]*final_output[weight_output]*(1-final_output[weight_output]);
					weight[i] -= learnrate * temp_gradient;

					weight_output++;
					weight_hide_to_output += hide_num;

				}
				
				weight_input++;
				weight_input %= input_dim;
				if(weight_input == 0)
					weight_hide++;

			}
		}

		error = 0;

		for(int i=0; i<output_e.size(); ++i)
		{
			temp_weight = hide_num*output_dim;

			for(int j=0; j<hide_num; ++j)
				hide_output[j] = 0;
			for(int j=0; j<output_dim; ++j)
				final_output[j] = 0;

			for(int k=0; k<hide_num; ++k)
			{
				for(int j=0; j<input_dim; ++j)
				{
					hide_output[k] += weight[temp_weight++] * input[i][j];
				}

				hide_output[k] -= bias[k];
				hide_output[k] = 1.0*1/(1+exp(-hide_output[k]));
			}

			temp_weight = 0;

			for(int k=0; k<output_dim; ++k)
			{
				for(int j=0; j<hide_num; ++j)
				{
					final_output[k] += weight[temp_weight++] * hide_output[j];
				}

				final_output[k] -= bias[hide_num + k];
				final_output[k] = 1.0*1/(1+exp(-final_output[k]));
			}

			

			for(int j=0; j<output_dim; ++j)
				error += (output_e[i][j] - final_output[j]) * (output_e[i][j] - final_output[j]);

		}

		validation_error = 0;
		for(int i=0; i<v_output_e.size(); ++i)
		{
			temp_weight = hide_num*output_dim;

			for(int j=0; j<hide_num; ++j)
				hide_output[j] = 0;
			for(int j=0; j<output_dim; ++j)
				final_output[j] = 0;

			for(int k=0; k<hide_num; ++k)
			{
				for(int j=0; j<input_dim; ++j)
				{
					hide_output[k] += weight[temp_weight++] * v_input[i][j];
				}

				hide_output[k] -= bias[k];
				hide_output[k] = 1.0*1/(1+exp(-hide_output[k]));
			}

			temp_weight = 0;

			for(int k=0; k<output_dim; ++k)
			{
				for(int j=0; j<hide_num; ++j)
				{
					final_output[k] += weight[temp_weight++] * hide_output[j];
				}

				final_output[k] -= bias[hide_num + k];
				final_output[k] = 1.0*1/(1+exp(-final_output[k]));
			}

			

			for(int j=0; j<output_dim; ++j)
				validation_error += (v_output_e[i][j] - final_output[j]) * (v_output_e[i][j] - final_output[j]);

		}
				
		error /= input.size();
		validation_error /= v_input.size();

		if(validation_error < min_verror) 
		{
			min_verror = validation_error;
			notimprove_cnt = 0;
		}
		else
			notimprove_cnt++;

		if(notimprove_cnt > 50000) break;



		rand_train_data = rand()%input.size();
		trainnum++;

			for(int j=0; j<hide_num; ++j)
				hide_output[j] = 0;
			for(int j=0; j<output_dim; ++j)
				final_output[j] = 0;

		cout<<trainnum<<" "<<error<<endl;

		//if(error > preerror)
		//	learnrate -= 0.0001;
		//else if((preerror - error) / preerror < 0.1)
		//	learnrate += 0.0001;

		//preerror = error;

	}

	double precision = 0;
	input.clear();
	output_e.clear();
	
	///test
	NN_data_preproces(test_data, input, output_e, input_dim, output_dim);
	data_nom_input(input);
	
	data_nom_output(output_e, isClassify_or_R);

	//for(int i=0; i<hide_output.size(); ++i)
	//	hide_output[i] = 0;
	//for(int i=0; i<final_output.size(); ++i)
	//	final_output[i] = 0;

	int position_weight;
	int temp_hide;
	bool isequal;

	for(int i=0; i<input.size(); ++i)
	{
		position_weight = hide_num*output_dim;
		temp_hide = 0;

		for(int j=0; j<hide_num; ++j)
				hide_output[j] = 0;
		for(int j=0; j<output_dim; ++j)
				final_output[j] = 0;

		for(int j=0; j<hide_num; ++j)
		{
			for(int k=0; k<input_dim; ++k)
			{
				hide_output[j] += weight[position_weight++]*input[i][k]; 
			}
			hide_output[j] -= bias[j];
			hide_output[j] = 1.0*1/(1+exp(-hide_output[j]));
		}

		position_weight = 0;

		for(int j=0; j<output_dim; ++j)
		{
			for(int k=0; k<hide_num; ++k)
			{
				final_output[j] += weight[position_weight++]*hide_output[k];
			}		
			final_output[j] -= bias[hide_num + j];
			final_output[j] = 1.0*1/(1+exp(-final_output[j]));
		}

		for(int j=0; j<output_dim; ++j)
			if(isClassify_or_R)
			{
				if(abs(final_output[j]-output_e[i][j]) < 1.0/kClassify/2)
					isequal = true;
				else
					isequal = false;
			}
			else
			{
				if(final_output[j]-output_e[i][j] < allowed_error)
					isequal = true;
				else
					isequal = false;
			}

		if(isequal)
			precision++;

		cout<<final_output[0]<<"  "<<output_e[i][0]<<endl;
	}

	precision /= input.size();

	//for(int i=0; i<(output_dim + input_dim)*hide_num; ++i)
	//	cout<<"weight["<<i<<"] = "<<weight[i]<<endl;

	//for(int i=0; i<output_dim+hide_num; ++i)
	//	cout<<"bias["<<i<<"] = "<<bias[i]<<endl;

	cout<<"precision is "<<precision<<endl;


}

int main(void)  
{  

	NN_train("E:\\ÑîÚS\\Code\\IRIS_TRAINDATA.txt", "E:\\ÑîÚS\\Code\\IRIS_TESTDATA.txt", "E:\\ÑîÚS\\Code\\IRIS_v.txt", 5, 4, 1, 0.5, 1, 30000, 0.4);
	//NN_train_("D:\\IRIS_TRAINDATA.txt", "D:\\IRIS_TESTDATA.txt", 0.1, 5, 4, 1, 0.01);
 return 0;  
}  

