// (c) Meta Platforms, Inc. and affiliates.

#import "output_redirect.h"

#import <cstdio>
#import <fstream>
#import <iostream>
#import <string>

#import <Foundation/Foundation.h>

class STDIORedirector {
 public:
  STDIORedirector() {
    if (@available(iOS 17, *)) {
      /* duplicate stdout */
      std::string file_name =
          std::string(std::getenv("HOME")) + "/tmp/BENCH_LOG";
      redirect_out_ = fopen(file_name.c_str(), "w");
      stdout_dupfd_ = dup(STDOUT_FILENO);
      stderr_dupfd_ = dup(STDERR_FILENO);
      /* replace stdout with our output fd */
      dup2(fileno(redirect_out_), STDOUT_FILENO);
      dup2(fileno(redirect_out_), STDERR_FILENO);
      fflush(stdout);
      fflush(stderr);
      setvbuf(stdout, nil, _IONBF, 0);
      setvbuf(stderr, nil, _IONBF, 0);
      setvbuf(redirect_out_, nil, _IONBF, 0);
    }
  }

  ~STDIORedirector() {
    if (@available(iOS 17, *)) {
      fflush(stdout);
      fflush(stderr);
      /* restore stdout */
      dup2(stdout_dupfd_, STDOUT_FILENO);
      dup2(stderr_dupfd_, STDERR_FILENO);
      close(stdout_dupfd_);
      close(stderr_dupfd_);
      fclose(redirect_out_);
    }
  }

 private:
  FILE* redirect_out_;
  int stdout_dupfd_;
  int stderr_dupfd_;
};

static STDIORedirector stdio_redirector_;
