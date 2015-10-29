#include "objects.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include<cmath>
#include<ctime>
#include<iostream>
#include<fstream>

////// SUPPORT CODE ///////
namespace {
 	
 inline unsigned char _brian_mod(unsigned char ux, unsigned char uy)
 {
     const unsigned char x = (unsigned char)ux;
     const unsigned char y = (unsigned char)uy;
     return ((x%y)+y)%y;
 }
 inline char _brian_mod(unsigned char ux, char uy)
 {
     const char x = (char)ux;
     const char y = (char)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned short _brian_mod(unsigned char ux, unsigned short uy)
 {
     const unsigned short x = (unsigned short)ux;
     const unsigned short y = (unsigned short)uy;
     return ((x%y)+y)%y;
 }
 inline short _brian_mod(unsigned char ux, short uy)
 {
     const short x = (short)ux;
     const short y = (short)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned int _brian_mod(unsigned char ux, unsigned int uy)
 {
     const unsigned int x = (unsigned int)ux;
     const unsigned int y = (unsigned int)uy;
     return ((x%y)+y)%y;
 }
 inline int _brian_mod(unsigned char ux, int uy)
 {
     const int x = (int)ux;
     const int y = (int)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long _brian_mod(unsigned char ux, unsigned long uy)
 {
     const unsigned long x = (unsigned long)ux;
     const unsigned long y = (unsigned long)uy;
     return ((x%y)+y)%y;
 }
 inline long _brian_mod(unsigned char ux, long uy)
 {
     const long x = (long)ux;
     const long y = (long)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long long _brian_mod(unsigned char ux, unsigned long long uy)
 {
     const unsigned long long x = (unsigned long long)ux;
     const unsigned long long y = (unsigned long long)uy;
     return ((x%y)+y)%y;
 }
 inline long long _brian_mod(unsigned char ux, long long uy)
 {
     const long long x = (long long)ux;
     const long long y = (long long)uy;
     return ((x%y)+y)%y;
 }
 inline float _brian_mod(unsigned char ux, float uy)
 {
     const float x = (float)ux;
     const float y = (float)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(unsigned char ux, double uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(unsigned char ux, long double uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline char _brian_mod(char ux, unsigned char uy)
 {
     const char x = (char)ux;
     const char y = (char)uy;
     return ((x%y)+y)%y;
 }
 inline char _brian_mod(char ux, char uy)
 {
     const char x = (char)ux;
     const char y = (char)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned short _brian_mod(char ux, unsigned short uy)
 {
     const unsigned short x = (unsigned short)ux;
     const unsigned short y = (unsigned short)uy;
     return ((x%y)+y)%y;
 }
 inline short _brian_mod(char ux, short uy)
 {
     const short x = (short)ux;
     const short y = (short)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned int _brian_mod(char ux, unsigned int uy)
 {
     const unsigned int x = (unsigned int)ux;
     const unsigned int y = (unsigned int)uy;
     return ((x%y)+y)%y;
 }
 inline int _brian_mod(char ux, int uy)
 {
     const int x = (int)ux;
     const int y = (int)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long _brian_mod(char ux, unsigned long uy)
 {
     const unsigned long x = (unsigned long)ux;
     const unsigned long y = (unsigned long)uy;
     return ((x%y)+y)%y;
 }
 inline long _brian_mod(char ux, long uy)
 {
     const long x = (long)ux;
     const long y = (long)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long long _brian_mod(char ux, unsigned long long uy)
 {
     const unsigned long long x = (unsigned long long)ux;
     const unsigned long long y = (unsigned long long)uy;
     return ((x%y)+y)%y;
 }
 inline long long _brian_mod(char ux, long long uy)
 {
     const long long x = (long long)ux;
     const long long y = (long long)uy;
     return ((x%y)+y)%y;
 }
 inline float _brian_mod(char ux, float uy)
 {
     const float x = (float)ux;
     const float y = (float)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(char ux, double uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(char ux, long double uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline unsigned short _brian_mod(unsigned short ux, unsigned char uy)
 {
     const unsigned short x = (unsigned short)ux;
     const unsigned short y = (unsigned short)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned short _brian_mod(unsigned short ux, char uy)
 {
     const unsigned short x = (unsigned short)ux;
     const unsigned short y = (unsigned short)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned short _brian_mod(unsigned short ux, unsigned short uy)
 {
     const unsigned short x = (unsigned short)ux;
     const unsigned short y = (unsigned short)uy;
     return ((x%y)+y)%y;
 }
 inline short _brian_mod(unsigned short ux, short uy)
 {
     const short x = (short)ux;
     const short y = (short)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned int _brian_mod(unsigned short ux, unsigned int uy)
 {
     const unsigned int x = (unsigned int)ux;
     const unsigned int y = (unsigned int)uy;
     return ((x%y)+y)%y;
 }
 inline int _brian_mod(unsigned short ux, int uy)
 {
     const int x = (int)ux;
     const int y = (int)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long _brian_mod(unsigned short ux, unsigned long uy)
 {
     const unsigned long x = (unsigned long)ux;
     const unsigned long y = (unsigned long)uy;
     return ((x%y)+y)%y;
 }
 inline long _brian_mod(unsigned short ux, long uy)
 {
     const long x = (long)ux;
     const long y = (long)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long long _brian_mod(unsigned short ux, unsigned long long uy)
 {
     const unsigned long long x = (unsigned long long)ux;
     const unsigned long long y = (unsigned long long)uy;
     return ((x%y)+y)%y;
 }
 inline long long _brian_mod(unsigned short ux, long long uy)
 {
     const long long x = (long long)ux;
     const long long y = (long long)uy;
     return ((x%y)+y)%y;
 }
 inline float _brian_mod(unsigned short ux, float uy)
 {
     const float x = (float)ux;
     const float y = (float)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(unsigned short ux, double uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(unsigned short ux, long double uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline short _brian_mod(short ux, unsigned char uy)
 {
     const short x = (short)ux;
     const short y = (short)uy;
     return ((x%y)+y)%y;
 }
 inline short _brian_mod(short ux, char uy)
 {
     const short x = (short)ux;
     const short y = (short)uy;
     return ((x%y)+y)%y;
 }
 inline short _brian_mod(short ux, unsigned short uy)
 {
     const short x = (short)ux;
     const short y = (short)uy;
     return ((x%y)+y)%y;
 }
 inline short _brian_mod(short ux, short uy)
 {
     const short x = (short)ux;
     const short y = (short)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned int _brian_mod(short ux, unsigned int uy)
 {
     const unsigned int x = (unsigned int)ux;
     const unsigned int y = (unsigned int)uy;
     return ((x%y)+y)%y;
 }
 inline int _brian_mod(short ux, int uy)
 {
     const int x = (int)ux;
     const int y = (int)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long _brian_mod(short ux, unsigned long uy)
 {
     const unsigned long x = (unsigned long)ux;
     const unsigned long y = (unsigned long)uy;
     return ((x%y)+y)%y;
 }
 inline long _brian_mod(short ux, long uy)
 {
     const long x = (long)ux;
     const long y = (long)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long long _brian_mod(short ux, unsigned long long uy)
 {
     const unsigned long long x = (unsigned long long)ux;
     const unsigned long long y = (unsigned long long)uy;
     return ((x%y)+y)%y;
 }
 inline long long _brian_mod(short ux, long long uy)
 {
     const long long x = (long long)ux;
     const long long y = (long long)uy;
     return ((x%y)+y)%y;
 }
 inline float _brian_mod(short ux, float uy)
 {
     const float x = (float)ux;
     const float y = (float)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(short ux, double uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(short ux, long double uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline unsigned int _brian_mod(unsigned int ux, unsigned char uy)
 {
     const unsigned int x = (unsigned int)ux;
     const unsigned int y = (unsigned int)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned int _brian_mod(unsigned int ux, char uy)
 {
     const unsigned int x = (unsigned int)ux;
     const unsigned int y = (unsigned int)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned int _brian_mod(unsigned int ux, unsigned short uy)
 {
     const unsigned int x = (unsigned int)ux;
     const unsigned int y = (unsigned int)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned int _brian_mod(unsigned int ux, short uy)
 {
     const unsigned int x = (unsigned int)ux;
     const unsigned int y = (unsigned int)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned int _brian_mod(unsigned int ux, unsigned int uy)
 {
     const unsigned int x = (unsigned int)ux;
     const unsigned int y = (unsigned int)uy;
     return ((x%y)+y)%y;
 }
 inline int _brian_mod(unsigned int ux, int uy)
 {
     const int x = (int)ux;
     const int y = (int)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long _brian_mod(unsigned int ux, unsigned long uy)
 {
     const unsigned long x = (unsigned long)ux;
     const unsigned long y = (unsigned long)uy;
     return ((x%y)+y)%y;
 }
 inline long _brian_mod(unsigned int ux, long uy)
 {
     const long x = (long)ux;
     const long y = (long)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long long _brian_mod(unsigned int ux, unsigned long long uy)
 {
     const unsigned long long x = (unsigned long long)ux;
     const unsigned long long y = (unsigned long long)uy;
     return ((x%y)+y)%y;
 }
 inline long long _brian_mod(unsigned int ux, long long uy)
 {
     const long long x = (long long)ux;
     const long long y = (long long)uy;
     return ((x%y)+y)%y;
 }
 inline float _brian_mod(unsigned int ux, float uy)
 {
     const float x = (float)ux;
     const float y = (float)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(unsigned int ux, double uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(unsigned int ux, long double uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline int _brian_mod(int ux, unsigned char uy)
 {
     const int x = (int)ux;
     const int y = (int)uy;
     return ((x%y)+y)%y;
 }
 inline int _brian_mod(int ux, char uy)
 {
     const int x = (int)ux;
     const int y = (int)uy;
     return ((x%y)+y)%y;
 }
 inline int _brian_mod(int ux, unsigned short uy)
 {
     const int x = (int)ux;
     const int y = (int)uy;
     return ((x%y)+y)%y;
 }
 inline int _brian_mod(int ux, short uy)
 {
     const int x = (int)ux;
     const int y = (int)uy;
     return ((x%y)+y)%y;
 }
 inline int _brian_mod(int ux, unsigned int uy)
 {
     const int x = (int)ux;
     const int y = (int)uy;
     return ((x%y)+y)%y;
 }
 inline int _brian_mod(int ux, int uy)
 {
     const int x = (int)ux;
     const int y = (int)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long _brian_mod(int ux, unsigned long uy)
 {
     const unsigned long x = (unsigned long)ux;
     const unsigned long y = (unsigned long)uy;
     return ((x%y)+y)%y;
 }
 inline long _brian_mod(int ux, long uy)
 {
     const long x = (long)ux;
     const long y = (long)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long long _brian_mod(int ux, unsigned long long uy)
 {
     const unsigned long long x = (unsigned long long)ux;
     const unsigned long long y = (unsigned long long)uy;
     return ((x%y)+y)%y;
 }
 inline long long _brian_mod(int ux, long long uy)
 {
     const long long x = (long long)ux;
     const long long y = (long long)uy;
     return ((x%y)+y)%y;
 }
 inline float _brian_mod(int ux, float uy)
 {
     const float x = (float)ux;
     const float y = (float)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(int ux, double uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(int ux, long double uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline unsigned long _brian_mod(unsigned long ux, unsigned char uy)
 {
     const unsigned long x = (unsigned long)ux;
     const unsigned long y = (unsigned long)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long _brian_mod(unsigned long ux, char uy)
 {
     const unsigned long x = (unsigned long)ux;
     const unsigned long y = (unsigned long)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long _brian_mod(unsigned long ux, unsigned short uy)
 {
     const unsigned long x = (unsigned long)ux;
     const unsigned long y = (unsigned long)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long _brian_mod(unsigned long ux, short uy)
 {
     const unsigned long x = (unsigned long)ux;
     const unsigned long y = (unsigned long)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long _brian_mod(unsigned long ux, unsigned int uy)
 {
     const unsigned long x = (unsigned long)ux;
     const unsigned long y = (unsigned long)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long _brian_mod(unsigned long ux, int uy)
 {
     const unsigned long x = (unsigned long)ux;
     const unsigned long y = (unsigned long)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long _brian_mod(unsigned long ux, unsigned long uy)
 {
     const unsigned long x = (unsigned long)ux;
     const unsigned long y = (unsigned long)uy;
     return ((x%y)+y)%y;
 }
 inline long _brian_mod(unsigned long ux, long uy)
 {
     const long x = (long)ux;
     const long y = (long)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long long _brian_mod(unsigned long ux, unsigned long long uy)
 {
     const unsigned long long x = (unsigned long long)ux;
     const unsigned long long y = (unsigned long long)uy;
     return ((x%y)+y)%y;
 }
 inline long long _brian_mod(unsigned long ux, long long uy)
 {
     const long long x = (long long)ux;
     const long long y = (long long)uy;
     return ((x%y)+y)%y;
 }
 inline float _brian_mod(unsigned long ux, float uy)
 {
     const float x = (float)ux;
     const float y = (float)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(unsigned long ux, double uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(unsigned long ux, long double uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long _brian_mod(long ux, unsigned char uy)
 {
     const long x = (long)ux;
     const long y = (long)uy;
     return ((x%y)+y)%y;
 }
 inline long _brian_mod(long ux, char uy)
 {
     const long x = (long)ux;
     const long y = (long)uy;
     return ((x%y)+y)%y;
 }
 inline long _brian_mod(long ux, unsigned short uy)
 {
     const long x = (long)ux;
     const long y = (long)uy;
     return ((x%y)+y)%y;
 }
 inline long _brian_mod(long ux, short uy)
 {
     const long x = (long)ux;
     const long y = (long)uy;
     return ((x%y)+y)%y;
 }
 inline long _brian_mod(long ux, unsigned int uy)
 {
     const long x = (long)ux;
     const long y = (long)uy;
     return ((x%y)+y)%y;
 }
 inline long _brian_mod(long ux, int uy)
 {
     const long x = (long)ux;
     const long y = (long)uy;
     return ((x%y)+y)%y;
 }
 inline long _brian_mod(long ux, unsigned long uy)
 {
     const long x = (long)ux;
     const long y = (long)uy;
     return ((x%y)+y)%y;
 }
 inline long _brian_mod(long ux, long uy)
 {
     const long x = (long)ux;
     const long y = (long)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long long _brian_mod(long ux, unsigned long long uy)
 {
     const unsigned long long x = (unsigned long long)ux;
     const unsigned long long y = (unsigned long long)uy;
     return ((x%y)+y)%y;
 }
 inline long long _brian_mod(long ux, long long uy)
 {
     const long long x = (long long)ux;
     const long long y = (long long)uy;
     return ((x%y)+y)%y;
 }
 inline float _brian_mod(long ux, float uy)
 {
     const float x = (float)ux;
     const float y = (float)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(long ux, double uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(long ux, long double uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline unsigned long long _brian_mod(unsigned long long ux, unsigned char uy)
 {
     const unsigned long long x = (unsigned long long)ux;
     const unsigned long long y = (unsigned long long)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long long _brian_mod(unsigned long long ux, char uy)
 {
     const unsigned long long x = (unsigned long long)ux;
     const unsigned long long y = (unsigned long long)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long long _brian_mod(unsigned long long ux, unsigned short uy)
 {
     const unsigned long long x = (unsigned long long)ux;
     const unsigned long long y = (unsigned long long)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long long _brian_mod(unsigned long long ux, short uy)
 {
     const unsigned long long x = (unsigned long long)ux;
     const unsigned long long y = (unsigned long long)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long long _brian_mod(unsigned long long ux, unsigned int uy)
 {
     const unsigned long long x = (unsigned long long)ux;
     const unsigned long long y = (unsigned long long)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long long _brian_mod(unsigned long long ux, int uy)
 {
     const unsigned long long x = (unsigned long long)ux;
     const unsigned long long y = (unsigned long long)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long long _brian_mod(unsigned long long ux, unsigned long uy)
 {
     const unsigned long long x = (unsigned long long)ux;
     const unsigned long long y = (unsigned long long)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long long _brian_mod(unsigned long long ux, long uy)
 {
     const unsigned long long x = (unsigned long long)ux;
     const unsigned long long y = (unsigned long long)uy;
     return ((x%y)+y)%y;
 }
 inline unsigned long long _brian_mod(unsigned long long ux, unsigned long long uy)
 {
     const unsigned long long x = (unsigned long long)ux;
     const unsigned long long y = (unsigned long long)uy;
     return ((x%y)+y)%y;
 }
 inline long long _brian_mod(unsigned long long ux, long long uy)
 {
     const long long x = (long long)ux;
     const long long y = (long long)uy;
     return ((x%y)+y)%y;
 }
 inline float _brian_mod(unsigned long long ux, float uy)
 {
     const float x = (float)ux;
     const float y = (float)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(unsigned long long ux, double uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(unsigned long long ux, long double uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long long _brian_mod(long long ux, unsigned char uy)
 {
     const long long x = (long long)ux;
     const long long y = (long long)uy;
     return ((x%y)+y)%y;
 }
 inline long long _brian_mod(long long ux, char uy)
 {
     const long long x = (long long)ux;
     const long long y = (long long)uy;
     return ((x%y)+y)%y;
 }
 inline long long _brian_mod(long long ux, unsigned short uy)
 {
     const long long x = (long long)ux;
     const long long y = (long long)uy;
     return ((x%y)+y)%y;
 }
 inline long long _brian_mod(long long ux, short uy)
 {
     const long long x = (long long)ux;
     const long long y = (long long)uy;
     return ((x%y)+y)%y;
 }
 inline long long _brian_mod(long long ux, unsigned int uy)
 {
     const long long x = (long long)ux;
     const long long y = (long long)uy;
     return ((x%y)+y)%y;
 }
 inline long long _brian_mod(long long ux, int uy)
 {
     const long long x = (long long)ux;
     const long long y = (long long)uy;
     return ((x%y)+y)%y;
 }
 inline long long _brian_mod(long long ux, unsigned long uy)
 {
     const long long x = (long long)ux;
     const long long y = (long long)uy;
     return ((x%y)+y)%y;
 }
 inline long long _brian_mod(long long ux, long uy)
 {
     const long long x = (long long)ux;
     const long long y = (long long)uy;
     return ((x%y)+y)%y;
 }
 inline long long _brian_mod(long long ux, unsigned long long uy)
 {
     const long long x = (long long)ux;
     const long long y = (long long)uy;
     return ((x%y)+y)%y;
 }
 inline long long _brian_mod(long long ux, long long uy)
 {
     const long long x = (long long)ux;
     const long long y = (long long)uy;
     return ((x%y)+y)%y;
 }
 inline float _brian_mod(long long ux, float uy)
 {
     const float x = (float)ux;
     const float y = (float)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(long long ux, double uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(long long ux, long double uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline float _brian_mod(float ux, unsigned char uy)
 {
     const float x = (float)ux;
     const float y = (float)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline float _brian_mod(float ux, char uy)
 {
     const float x = (float)ux;
     const float y = (float)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline float _brian_mod(float ux, unsigned short uy)
 {
     const float x = (float)ux;
     const float y = (float)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline float _brian_mod(float ux, short uy)
 {
     const float x = (float)ux;
     const float y = (float)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline float _brian_mod(float ux, unsigned int uy)
 {
     const float x = (float)ux;
     const float y = (float)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline float _brian_mod(float ux, int uy)
 {
     const float x = (float)ux;
     const float y = (float)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline float _brian_mod(float ux, unsigned long uy)
 {
     const float x = (float)ux;
     const float y = (float)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline float _brian_mod(float ux, long uy)
 {
     const float x = (float)ux;
     const float y = (float)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline float _brian_mod(float ux, unsigned long long uy)
 {
     const float x = (float)ux;
     const float y = (float)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline float _brian_mod(float ux, long long uy)
 {
     const float x = (float)ux;
     const float y = (float)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline float _brian_mod(float ux, float uy)
 {
     const float x = (float)ux;
     const float y = (float)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(float ux, double uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(float ux, long double uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(double ux, unsigned char uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(double ux, char uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(double ux, unsigned short uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(double ux, short uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(double ux, unsigned int uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(double ux, int uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(double ux, unsigned long uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(double ux, long uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(double ux, unsigned long long uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(double ux, long long uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(double ux, float uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline double _brian_mod(double ux, double uy)
 {
     const double x = (double)ux;
     const double y = (double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(double ux, long double uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(long double ux, unsigned char uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(long double ux, char uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(long double ux, unsigned short uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(long double ux, short uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(long double ux, unsigned int uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(long double ux, int uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(long double ux, unsigned long uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(long double ux, long uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(long double ux, unsigned long long uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(long double ux, long long uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(long double ux, float uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(long double ux, double uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }
 inline long double _brian_mod(long double ux, long double uy)
 {
     const long double x = (long double)ux;
     const long double y = (long double)uy;
     return fmod(fmod(x, y)+y, y);
 }

}

////// HASH DEFINES ///////



void _run_neurongroup_stateupdater_codeobject()
{	
	using namespace brian;

    const double _start_time = omp_get_wtime();

	///// CONSTANTS ///////////
	const int _numg = 100;
const int _numt = 1;
const int _numv = 100;
const int _numdt = 1;
const int _numlastspike = 100;
const int _numnot_refractory = 100;
	///// POINTERS ////////////
 	
 double* __restrict  _ptr_array_neurongroup_g = _array_neurongroup_g;
 double* __restrict  _ptr_array_defaultclock_t = _array_defaultclock_t;
 double* __restrict  _ptr_array_neurongroup_v = _array_neurongroup_v;
 double* __restrict  _ptr_array_defaultclock_dt = _array_defaultclock_dt;
 double* __restrict  _ptr_array_neurongroup_lastspike = _array_neurongroup_lastspike;
 char* __restrict  _ptr_array_neurongroup_not_refractory = _array_neurongroup_not_refractory;


	//// MAIN CODE ////////////
	// scalar code
	const int _vectorisation_idx = -1;
 	
 const double dt = _ptr_array_defaultclock_dt[0];
 const double _lio_const_1 = - dt;


	const int _N = 100;
	#pragma omp parallel for schedule(static)
	for(int _idx=0; _idx<_N; _idx++)
	{
	    // vector code
		const int _vectorisation_idx = _idx;
                
        const double dt = _ptr_array_defaultclock_dt[0];
        const double lastspike = _ptr_array_neurongroup_lastspike[_idx];
        const double t = _ptr_array_defaultclock_t[0];
        double g = _ptr_array_neurongroup_g[_idx];
        double v = _ptr_array_neurongroup_v[_idx];
        char not_refractory;
        not_refractory = (t - lastspike) > 0.005;
        const double _g = ((_lio_const_1 * g) / 0.005) + g;
        const double _v = ((dt * (((-0.049) + g) - v)) / 0.02) + v;
        g = _g;
        v = _v;
        _ptr_array_neurongroup_not_refractory[_idx] = not_refractory;
        _ptr_array_neurongroup_g[_idx] = g;
        _ptr_array_neurongroup_v[_idx] = v;

	}

    const double _run_time = omp_get_wtime() -_start_time;
    neurongroup_stateupdater_codeobject_profiling_info += _run_time;
}


