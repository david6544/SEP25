#pragma once

#include <string>
#include <ranges>
#include <iostream>

namespace skx::io {

using Logger = int;

template <typename T>
void appendValToString(std::string& buf, T val, const std::string& prefix, const std::string& postfix) {
    buf += prefix;
    buf += std::to_string(val);
    buf += postfix;
}

template <typename T, std::ranges::input_range Range>
    requires std::convertible_to<std::ranges::range_value_t<Range>, T>
void appendValRangeToString(std::string& buf, const Range& range, const std::string& seperator, const std::string& prefix, const std::string& postfix) {
    buf += prefix;
    bool first = true;
    std::string stval;
    stval.reserve(1 + seperator.size() + prefix.size() + postfix.size());
    for (const std::ranges::range_value_t<Range>& val : range) {
        stval.clear();
        stval += seperator;
        if (true == first) {
            stval.clear();
            first = false;
        }
        stval += std::to_string(static_cast<T>(val));
        buf += stval;
    }
    buf += postfix;
}

Logger createLogger() {
    static int c = 0;
    return Logger{c++};
}

Logger& getDefaultLoggerRef() {
    static Logger logger = 0;// createLogger();
    return logger;
}

Logger& getDefaultErrorLoggerRef() {
    static Logger error_logger = 1;// createLogger();
    return error_logger;
}

void dissableLogger(Logger& logger) {
    logger = -1;
}

void enableLogger(Logger& logger) {
    logger = createLogger();
}

void log(const Logger& logger, const std::string& st) {
    const std::string ANSI_RESET = "\033[0m";
    const std::string ANSI_RED = "\033[31m";
    const std::string ANSI_GREEN = "\033[32m";
    const std::string ANSI_BLUE = "\033[34m";
    if (-1 == logger) {
        return;
    }
    else if (0 == logger) {
        std::cout << ANSI_GREEN;
    }
    else if (1 == logger) {
        std::cout << ANSI_RED;
    }
    std::cout << st << ANSI_RESET;
}


} // namespace skx::io
