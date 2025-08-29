#include <iostream>
#include <gtest/gtest.h>

#include "../src/InputOutput/InputOutput.hpp"
#include "../src/InputOutput/CommandLineInputOutput.hpp"


TEST(TestCommandLine, testCLIOinnit){
    // use clio input output
    CommandLineInputOutput::set_IO();

    InputOutput* io = InputOutput::get_instance();
    EXPECT_NE(io, nullptr);

    InputOutput* io2 = InputOutput::get_instance();
    EXPECT_EQ(io, io2);
}

// basic test of clio with 0d query
TEST(TestCommandLine, testingCLIO__1){
    // use clio input output
    CommandLineInputOutput::set_IO();

    InputOutput* io = InputOutput::get_instance();
    
    // Redirect cout to a stringstream
    std::stringstream buffer;
    std::streambuf* oldCout = std::cout.rdbuf(buffer.rdbuf());

    std::istringstream input("42\n");
    std::streambuf* oldCin = std::cin.rdbuf(input.rdbuf());

    std::vector<int> query = {0,0,0};
    double result = io->send_query_recieve_result(query);
    std::cout.rdbuf(oldCout);

    EXPECT_EQ(result, 42);
    EXPECT_EQ(buffer.str(), "0,0,0\n");
}

// test with query with non zero values
TEST(TestCommandLine, testingCLIO__2){
    // use clio input output
    CommandLineInputOutput::set_IO();

    InputOutput *io = InputOutput::get_instance();
    
    // Redirect cout to a stringstream
    std::stringstream buffer;
    std::streambuf* oldCout = std::cout.rdbuf(buffer.rdbuf());

    std::istringstream input("1.001\n");
    std::streambuf* oldCin = std::cin.rdbuf(input.rdbuf());

    std::vector<int> query = {1,2,3};
    double result = io->send_query_recieve_result(query);
    std::cout.rdbuf(oldCout);

    EXPECT_EQ(result, 1.001);
    EXPECT_EQ(buffer.str(), "1,2,3\n");
}

// test with query size of 1
TEST(TestCommandLine, testingCLIO__3){
    // use clio input output
    CommandLineInputOutput::set_IO();

    InputOutput* io = InputOutput::get_instance();
    
    // Redirect cout to a stringstream
    std::stringstream buffer;
    std::streambuf* oldCout = std::cout.rdbuf(buffer.rdbuf());

    std::istringstream input("1.001\n");
    std::streambuf* oldCin = std::cin.rdbuf(input.rdbuf());

    std::vector<int> query = {1};
    double result = io->send_query_recieve_result(query);
    std::cout.rdbuf(oldCout);

    EXPECT_EQ(result, 1.001);
    EXPECT_EQ(buffer.str(), "1\n");
}

// test expected when non num is input
TEST(TestCommandLine, testingCLIO__4){
    // use clio input output
    CommandLineInputOutput::set_IO();

    InputOutput* io = InputOutput::get_instance();
    
    // Redirect cout to a stringstream
    std::stringstream buffer;
    std::streambuf* oldCout = std::cout.rdbuf(buffer.rdbuf());

    std::istringstream input("test\n");
    std::streambuf* oldCin = std::cin.rdbuf(input.rdbuf());

    std::vector<int> query = {1,2,3};
    double result = io->send_query_recieve_result(query);
    std::cout.rdbuf(oldCout);

    EXPECT_EQ(result, 0);
    EXPECT_EQ(buffer.str(), "1,2,3\n");
}