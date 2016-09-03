#ifndef LINE_H
#define LINE_H

#include <opencv2/core.hpp>

using namespace cv;

class Line
{
	private:
		Point pt1, pt2;

		bool intersect(const Point& _pt1, const Point& _pt2) const;
		bool sameSigns(int a, int b) const;
	
	public:
		Line() = default;
		Line(int lineID, const Point& _pt1, const Point& _pt2);
		int ID;
		bool isBeingCrossed;

		bool intersect(const Rect& rect) const;
		void draw(InputOutputArray _frame);
};

#endif
