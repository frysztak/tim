#ifndef DIRECTION_H
#define DIRECTION_H

#include <string>
#include <map>

class Direction
{
    private:
        enum DirectionEnum
        {
            UP,
            DOWN,
            LEFT,
            RIGHT
        };

        DirectionEnum direction;
        uint32_t counter;

    public:
        Direction() = default;
        Direction(const DirectionEnum direction);
        Direction(const std::string& directionStr);
        Direction(const Direction& direction) = default;
        std::string prettyString();
        Direction operator!();
        Direction operator++(int);
        Direction& operator=(const Direction&& other);
        
        typedef std::map<DirectionEnum, std::string> Dictionary;
        static const Dictionary dictionary;
};

#endif

/* vim: set ft=cpp ts=4 sw=4 sts=4 tw=0 fenc=utf-8 et: */
