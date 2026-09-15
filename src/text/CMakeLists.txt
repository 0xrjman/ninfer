add_library(ninfer_text STATIC
  unicode.cpp
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc/utf8proc.c
)
ninfer_internal_includes(ninfer_text)
target_include_directories(ninfer_text PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
