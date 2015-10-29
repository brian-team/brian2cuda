#include "objects.h"
#include "code_objects/statemonitor_codeobject.h"
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



void _run_statemonitor_codeobject()
{	
	using namespace brian;

    const double _start_time = omp_get_wtime();

	///// CONSTANTS ///////////
	const int _num_indices = 10;
const int _num_clock_t = 1;
double* const _array_synapses_w = _dynamic_array_synapses_w.empty()? 0 : &_dynamic_array_synapses_w[0];
const int _num_source_w = _dynamic_array_synapses_w.size();
double* const _array_statemonitor_t = _dynamic_array_statemonitor_t.empty()? 0 : &_dynamic_array_statemonitor_t[0];
const int _numt = _dynamic_array_statemonitor_t.size();
const int _numN = 1;
	///// POINTERS ////////////
 	
 int32_t* __restrict  _ptr_array_statemonitor__indices = _array_statemonitor__indices;
 double* __restrict  _ptr_array_defaultclock_t = _array_defaultclock_t;
 double* __restrict  _ptr_array_synapses_w = _array_synapses_w;
 double* __restrict  _ptr_array_statemonitor_t = _array_statemonitor_t;
 int32_t* __restrict  _ptr_array_statemonitor_N = _array_statemonitor_N;



    _dynamic_array_statemonitor_t.push_back(_ptr_array_defaultclock_t[0]);

    const int _new_size = _dynamic_array_statemonitor_t.size();
    // Resize the dynamic arrays
    _dynamic_array_statemonitor_w.resize(_new_size, _num_indices);

    // scalar code
    const int _vectorisation_idx = -1;
        


    #pragma omp parallel for schedule(static)
    for (int _i = 0; _i < _num_indices; _i++)
    {
        // vector code
        const int _idx = _ptr_array_statemonitor__indices[_i];
        const int _vectorisation_idx = _idx;
                                        
                    const double _source_w = _ptr_array_synapses_w[_idx];
                    const double _to_record_w = _source_w;



            _dynamic_array_statemonitor_w(_new_size-1, _i) = _to_record_w;
    }

    _ptr_array_statemonitor_N[0] = _new_size;


    const double _run_time = omp_get_wtime() -_start_time;
    statemonitor_codeobject_profiling_info += _run_time;
}


