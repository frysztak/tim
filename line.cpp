#include "line.h"

Line::Line(int id, const Point& _pt1, const Point& _pt2) : pt1(_pt1), pt2(_pt2), ID(id)
{
}

bool Line::sameSigns(int a, int b) const
{
	if (a > 0 && b > 0)
		return true;
	if (a < 0 && b < 0)
		return true;
	if (a == 0 && b == 0)
		return true;
	return false;
}

bool Line::intersect(const Point& _pt1, const Point& _pt2) const
{
	// adapted from Graphics Gems volume II
	// https://webdocs.cs.ualberta.ca/~graphics/books/GraphicsGems/gemsii/xlines.c
	
	long a1, a2, b1, b2, c1, c2; /* Coefficients of line eqns. */
    long r1, r2, r3, r4;         /* 'Sign' values */
    long denom;				     /* Intermediate values */

    /* Compute a1, b1, c1, where line joining points 1 and 2
     * is "a1 x  +  b1 y  +  c1  =  0".  */

    a1 = pt2.y - pt1.y;
    b1 = pt1.x - pt2.x;
    c1 = pt2.x * pt1.y - pt1.x * pt2.y;

    /* Compute r3 and r4. */

    r3 = a1 * _pt1.x + b1 * _pt1.y + c1;
    r4 = a1 * _pt2.x + b1 * _pt2.y + c1;

    /* Check signs of r3 and r4.  If both point 3 and point 4 lie on
     * same side of line 1, the line segments do not intersect. */

    if (r3 != 0 && r4 != 0 && sameSigns(r3, r4))
        return false;

    /* Compute a2, b2, c2 */

    a2 = _pt2.y - _pt1.y;
    b2 = _pt1.x - _pt2.x;
    c2 = _pt2.x * _pt1.y - _pt1.x * _pt2.y;

    /* Compute r1 and r2 */

    r1 = a2 * pt1.x + b2 * pt1.y + c2;
    r2 = a2 * pt2.x + b2 * pt2.y + c2;

    /* Check signs of r1 and r2.  If both point 1 and point 2 lie
     * on same side of second line segment, the line segments do
     * not intersect. */

    if (r1 != 0 && r2 != 0 && sameSigns(r1, r2))
        return false;

    denom = a1 * b2 - a2 * b1;
    if (denom == 0)
        return false; // collinear
       
    return true;
}

bool Line::intersect(const Rect& rect) const
{
	auto x = rect.x, y = rect.y;
	auto width = rect.width, height = rect.height;

	std::vector<bool> intersections = 
	{
		intersect(Point(x, y), Point(x + width, y)),
		intersect(Point(x, y), Point(x, y + height)),
		intersect(Point(x + width, y + height), Point(x - width, y)),
		intersect(Point(x + width, y + height), Point(x, y + height))
	};

	return std::any_of(intersections.begin(), intersections.end(), [](bool i){ return i == true; });
}

void Line::draw(InputOutputArray _frame)
{
	Mat frame = _frame.getMat();
	Scalar lineColour = isBeingCrossed ? Scalar(19,38,242) : Scalar(197,247,200);

	line(frame, pt1, pt2, lineColour, 2, LINE_AA);
}
