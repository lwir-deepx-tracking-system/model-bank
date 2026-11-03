#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdint>
extern "C" void decode_raw(const float* box,const float* scores,float* out,int height,int width){
 const int N=(height/8)*(width/8)+(height/16)*(width/16)+(height/32)*(width/32);int offset=0;
 for(int stride: {8,16,32}){int w=width/stride,h=height/stride;
  for(int j=0;j<h*w;++j){int i=offset+j;float ax=float(j%w)+.5f,ay=float(j/w)+.5f;
   float l=ax-box[i],t=ay-box[N+i],r=ax+box[2*N+i],b=ay+box[3*N+i];
   out[i]=((l+r)*.5f)*stride;out[N+i]=((t+b)*.5f)*stride;
   out[2*N+i]=(r-l)*stride;out[3*N+i]=(b-t)*stride;
  }offset+=h*w;
 }
 // Sigmoid is performed by OpenCV SIMD in the caller.
}
struct Box {float x1,y1,x2,y2,score;int cls;};
extern "C" int nms_decoded(const float* in,float* out,int N){
 std::vector<Box> v;v.reserve(4000);
 for(int i=0;i<N;++i)for(int c=0;c<16;++c){float s=in[(c+4)*N+i];if(s>.001f){
  float x=in[i],y=in[N+i],w=in[2*N+i],h=in[3*N+i];
  v.push_back({x-w*.5f,y-h*.5f,x+w*.5f,y+h*.5f,s,c});
 }}
 std::stable_sort(v.begin(),v.end(),[](const Box&a,const Box&b){return a.score>b.score;});
 if(v.size()>30000)v.resize(30000);
 std::vector<uint8_t> suppressed(v.size(),0);
 for(int c=0;c<16;++c){std::vector<int> ids;for(size_t i=0;i<v.size();++i)if(v[i].cls==c)ids.push_back(int(i));
  for(size_t a=0;a<ids.size();++a){int i=ids[a];if(suppressed[i])continue;const auto& u=v[i];float area=(u.x2-u.x1)*(u.y2-u.y1);
   for(size_t b=a+1;b<ids.size();++b){int j=ids[b];if(suppressed[j])continue;const auto& z=v[j];
    float iw=std::max(0.f,std::min(u.x2,z.x2)-std::max(u.x1,z.x1));float ih=std::max(0.f,std::min(u.y2,z.y2)-std::max(u.y1,z.y1));
    float inter=iw*ih,other=(z.x2-z.x1)*(z.y2-z.y1);float ratio=inter/(area+other-inter);
    if(ratio>.7f)suppressed[j]=1;
   }
  }
 }
 int n=0;for(size_t i=0;i<v.size()&&n<300;++i)if(!suppressed[i]){auto& b=v[i];float* p=out+6*n++;p[0]=b.x1;p[1]=b.y1;p[2]=b.x2;p[3]=b.y2;p[4]=b.score;p[5]=float(b.cls);}
 return n;
}
