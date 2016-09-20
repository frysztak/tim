#include "direction.h"

const Direction::Dictionary Direction::dictionary = 
{
    { DirectionEnum::UP, "up" },
    { DirectionEnum::DOWN, "down" },
    { DirectionEnum::LEFT, "left" },
    { DirectionEnum::RIGHT, "right" }
};

Direction::Direction(const DirectionEnum direction) : direction(direction), counter(0)
{
}

Direction::Direction(const std::string& directionStr) : counter(0)
{
   for (auto it = dictionary.begin(); it != dictionary.end(); ++it)
       if (it->second == directionStr)
           this->direction = it->first;
}

std::string Direction::prettyString()
{
    return dictionary.at(direction) + ": " + std::to_string(counter);
}

Direction Direction::operator!()
{
    if (direction == DirectionEnum::UP)
        return Direction(DirectionEnum::DOWN);
    else if (direction == DirectionEnum::DOWN)
        return Direction(DirectionEnum::UP);
    else if (direction == DirectionEnum::LEFT)
        return Direction(Direction::RIGHT);
    else
        return Direction(Direction::LEFT);
}

Direction Direction::operator++(int)
{
    counter++;
    return *this;
}

Direction& Direction::operator=(const Direction&& other)
{
    this->direction = other.direction;
    this->counter = other.counter;
    return *this;
}

/* vim: set ft=cpp ts=4 sw=4 sts=4 tw=0 fenc=utf-8 et: */
