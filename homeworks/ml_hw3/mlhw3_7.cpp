#include<math.h>
#include<iostream>

using namespace std;

float E_u(float u,float v){
  float re= exp(u)+v*exp(u*v)+2*u-2*v-3;
  cout<<"   E_u result is "<<re<<endl;
  return re;
}

float E_v(float u, float v ){
  float re= 2*exp(2*v)+u*exp(u*v)-2*u+4*v-2;
  cout<<"   E_v result is "<<re<<endl;
  return re;
}

int main(){
  float u=0;
  float v=0;
  int n=0;
  while(n<5){
    cout<<n+1<<" th iteration"<<endl;
    float unew=u-0.01*E_u(u,v);
    float vnew=v-0.01*E_v(u,v);
    u=unew;
    v=vnew;
    cout<<"u is "<<u<<", v is "<<v<<endl;
    ++n;
  }
  u=0;
  v=0;
  cout<<"Final answer is "<<exp(u)+exp(2*v)+exp(u*v)+u*u-2*u*v+2*v*v-3*u-2*v<<endl;
}
