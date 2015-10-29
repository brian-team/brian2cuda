#include "objects.h"
#include "code_objects/spikemonitor_codeobject.h"
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



void _run_spikemonitor_codeobject()
{	
	using namespace brian;

    const double _start_time = omp_get_wtime();

	///// CONSTANTS ///////////
	const int _num_source_t = 1;
const int _numcount = 100;
const int _num_clock_t = 1;
const int _num_spikespace = 101;
int32_t* const _array_spikemonitor_i = _dynamic_array_spikemonitor_i.empty()? 0 : &_dynamic_array_spikemonitor_i[0];
const int _numi = _dynamic_array_spikemonitor_i.size();
const int _numN = 1;
double* const _array_spikemonitor_t = _dynamic_array_spikemonitor_t.empty()? 0 : &_dynamic_array_spikemonitor_t[0];
const int _numt = _dynamic_array_spikemonitor_t.size();
const int _num_source_idx = 100;
const int _num_source_i = 100;
	///// POINTERS ////////////
 	
 double* __restrict  _ptr_array_defaultclock_t = _array_defaultclock_t;
 int32_t* __restrict  _ptr_array_spikemonitor_count = _array_spikemonitor_count;
 int32_t* __restrict  _ptr_array_neurongroup__spikespace = _array_neurongroup__spikespace;
 int32_t* __restrict  _ptr_array_spikemonitor_i = _array_spikemonitor_i;
 int32_t* __restrict  _ptr_array_spikemonitor_N = _array_spikemonitor_N;
 double* __restrict  _ptr_array_spikemonitor_t = _array_spikemonitor_t;
 int32_t* __restrict  _ptr_array_spikemonitor__source_idx = _array_spikemonitor__source_idx;
 int32_t* __restrict  _ptr_array_neurongroup_i = _array_neurongroup_i;


	//// MAIN CODE ////////////

	int32_t _num_events = _ptr_array_neurongroup__spikespace[_num_spikespace-1];

    if (_num_events > 0)
    {
        int _start_idx = _num_events;
        int _end_idx = _num_events;
        for(int _j=0; _j<_num_events; _j++)
        {
            const int _idx = _ptr_array_neurongroup__spikespace[_j];
            if (_idx >= 0) {
                _start_idx = _j;
                break;
            }
        }
        for(int _j=_start_idx; _j<_num_events; _j++)
        {
            const int _idx = _ptr_array_neurongroup__spikespace[_j];
            if (_idx >= 100) {
                _end_idx = _j;
                break;
            }
        }
        _num_events = _end_idx - _start_idx;
        if (_num_events > 0) {
             const int _vectorisation_idx = 1;
                        

            for(int _j=_start_idx; _j<_end_idx; _j++)
            {
                const int _idx = _ptr_array_neurongroup__spikespace[_j];
                const int _vectorisation_idx = _idx;
                                
                const double _source_t = _ptr_array_defaultclock_t[0];
                const int32_t _source_i = _ptr_array_neurongroup_i[_idx];
                const int32_t _to_record_i = _source_i;
                const double _to_record_t = _source_t;

                _dynamic_array_spikemonitor_i.push_back(_to_record_i);
                _dynamic_array_spikemonitor_t.push_back(_to_record_t);
                _ptr_array_spikemonitor_count[_idx-0]++;
            }
            _ptr_array_spikemonitor_N[0] += _num_events;
        }
    }


    const double _run_time = omp_get_wtime() -_start_time;
    spikemonitor_codeobject_profiling_info += _run_time;
}

void _debugmsg_spikemonitor_codeobject()
{
	using namespace brian;
    const int _num_source_t = 1;
const int _numcount = 100;
const int _num_clock_t = 1;
const int _num_spikespace = 101;
int32_t* const _array_spikemonitor_i = _dynamic_array_spikemonitor_i.empty()? 0 : &_dynamic_array_spikemonitor_i[0];
const int _numi = _dynamic_array_spikemonitor_i.size();
const int _numN = 1;
double* const _array_spikemonitor_t = _dynamic_array_spikemonitor_t.empty()? 0 : &_dynamic_array_spikemonitor_t[0];
const int _numt = _dynamic_array_spikemonitor_t.size();
const int _num_source_idx = 100;
const int _num_source_i = 100;
        
    double* __restrict  _ptr_array_defaultclock_t = _array_defaultclock_t;
    int32_t* __restrict  _ptr_array_spikemonitor_count = _array_spikemonitor_count;
    int32_t* __restrict  _ptr_array_neurongroup__spikespace = _array_neurongroup__spikespace;
    int32_t* __restrict  _ptr_array_spikemonitor_i = _array_spikemonitor_i;
    int32_t* __restrict  _ptr_array_spikemonitor_N = _array_spikemonitor_N;
    double* __restrict  _ptr_array_spikemonitor_t = _array_spikemonitor_t;
    int32_t* __restrict  _ptr_array_spikemonitor__source_idx = _array_spikemonitor__source_idx;
    int32_t* __restrict  _ptr_array_neurongroup_i = _array_neurongroup_i;

	std::cout << "Number of spikes: " << _ptr_array_spikemonitor_N[0] << endl;
}

