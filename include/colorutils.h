#ifndef COLORUTILS_H
#define COLORUTILS_H

#include "platform.h"

static vec4 hsv2rgb(REAL h, REAL s, REAL v)
{
    int i = (int) std::floor(h);
    REAL f = h - i;
    if(!(i & 0x1))
        f = (REAL)1.0 - f;
    REAL m = v * ((REAL)1.0 - s);
    REAL n = v * ((REAL)1.0 - s * f);
    switch(i)
    {
    case 6:
    case 0:
        return vec4(v,n,m,1);
    case 1:
        return vec4(n,v,m,1);
    case 2:
        return vec4(m,v,n,1);
    case 3:
        return vec4(m,n,v,1);
    case 4:
        return vec4(n,m,v,1);
    case 5:
        return vec4(v,m,n,1);
    }

    return vec4(0,0,0,1);
}

/*! colorByValue()
 *
 *  \brief  given a real-valued scalar a color is generated from blue to red
 *          the values in _I are truncated at threshold first
 **/
static vec4 colorByValue(REAL I, REAL threshold = REAL(1))
{
    REAL curR = std::max(I, REAL(0));
    curR = std::min(curR, threshold);
    curR /= threshold;
    REAL x = (REAL(1) - curR)*REAL(2);
    return hsv2rgb(x, (REAL)0.9, (REAL)0.9);
}

/*! colorByValue()
 *
 *  \brief  given a real-valued scalar a color is generated from blue to red
 *          the values in _I are truncated at threshold first
 **/
static void colorByValue(const std::vector<REAL>& _I, REAL threshold, std::vector<vec4>& colorOut)
{
    colorOut.clear();

    for(unsigned int i = 0; i < _I.size(); ++i)
    {
        REAL curR = std::min(_I[i], threshold);
        curR /= threshold;
        REAL x = (REAL(1) - curR)*REAL(2);
        colorOut.push_back(hsv2rgb(x, (REAL)0.9, (REAL)0.9));
    }
}

#endif // COLORUTILS_H
