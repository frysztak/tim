#ifndef LINE_H
#define LINE_H

#include "movingobject.h"

using namespace cv;

class Line
{
	private:
		Point pt1, pt2;
		bool isBeingCrossed;
		bool intersect(const Point& _pt1, const Point& _pt2) const;
		bool intersect(const Rect& rect) const;
		bool sameSigns(int a, int b) const;
	
	public:
		Line() = default;
		Line(const Point& _pt1, const Point& _pt2);

		void intersect(const std::vector<MovingObject>& objects);
		void draw(InputOutputArray _frame);
};

#endif
