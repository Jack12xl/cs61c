
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "utils.h"
#include "tables.h"

const int SYMTBL_NON_UNIQUE = 0;
const int SYMTBL_UNIQUE_NAME = 1;

/*******************************
 * Helper Functions
 *******************************/

void allocation_failed() {
    write_to_log("Error: allocation failed\n");
    exit(1);
}

void addr_alignment_incorrect() {
    write_to_log("Error: address is not a multiple of 4.\n");
}

void name_already_exists(const char* name) {
    write_to_log("Error: name '%s' already exists in table.\n", name);
}

void write_symbol(FILE* output, uint32_t addr, const char* name) {
    fprintf(output, "%u\t%s\n", addr, name);
}

/*******************************
 * Symbol Table Functions
 *******************************/

/* Creates a new SymbolTable containg 0 elements and returns a pointer to that
   table. Multiple SymbolTables may exist at the same time. 
   If memory allocation fails, you should call allocation_failed(). 
   Mode will be either SYMTBL_NON_UNIQUE or SYMTBL_UNIQUE_NAME. You will need
   to store this value for use during add_to_table().
 */
SymbolTable* create_table(int mode) {
    SymbolTable* table = (SymbolTable*) malloc(sizeof(SymbolTable));
    if (!table) {
      allocation_failed();
    }
    table->mode = mode;
    table->len = 0;
    table->cap = 8;
    table->tbl = (Symbol*) malloc(table->cap * sizeof(Symbol));
    return table;
}

/* Frees the given SymbolTable and all associated memory. */
void free_table(SymbolTable* table) {
    if (!table) {
      return;
    }
    free(table->tbl);
    free(table);
}

/* Adds a new symbol and its address to the SymbolTable pointed to by TABLE. 
   ADDR is given as the byte offset from the first instruction. The SymbolTable
   must be able to resize itself as more elements are added. 

   Note that NAME may point to a temporary array, so it is not safe to simply
   store the NAME pointer. You must store a copy of the given string.

   If ADDR is not word-aligned, you should call addr_alignment_incorrect() and
   return -1. If the table's mode is SYMTBL_UNIQUE_NAME and NAME already exists 
   in the table, you should call name_already_exists() and return -1. If memory
   allocation fails, you should call allocation_failed(). 

   Otherwise, you should store the symbol name and address and return 0.
 */
int add_to_table(SymbolTable* table, const char* name, uint32_t addr) {
    if (!table) {
      allocation_failed();
    }

    if (addr % 4 != 0) {
      addr_alignment_incorrect();
      return -1;
    }

    // If the table len is at the cap, need to resize.
    if (table->len == table->cap) {
      Symbol* newtable = (Symbol*) malloc(2 * table->cap * sizeof(Symbol));
      if (!newtable) {
        allocation_failed();
      }
      for (int i = 0; i < table->len; i++) {
        newtable[i] = (table->tbl)[i];
      }
      free(table->tbl);
      table->cap *= 2;
      table->tbl = newtable;
    }

    Symbol* current = table->tbl;
    if (!current) {
        allocation_failed();
    } else {
      // Check for already assigned names if mode is SYMTBL_UNIQUE_NAME.
      if (table->mode == SYMTBL_UNIQUE_NAME) {
        for (int i = 0; i < table->len; i++) {
          current = &((table->tbl)[i]);
          if (strcmp(current->name, name) == 0) {
            name_already_exists(name);
            return -1;
          }
        }
      }
      // At this point name is OK to add. Create a new symbol.
      char* sym_name = (char *) malloc(strlen(name) + 1);
      strcpy(sym_name, name);
      (table->tbl[table->len]).name = sym_name;
      (table->tbl[table->len]).addr = addr;
      // Add the symbol to the table. Increment the length of the table.
      table->len += 1;

    }

    return 0;
}

/* Returns the address (byte offset) of the given symbol. If a symbol with name
   NAME is not present in TABLE, return -1.
 */
int64_t get_addr_for_symbol(SymbolTable* table, const char* name) {
    if (!table) {
        return -1;
    } else {
      Symbol* current;
      // Check for already assigned names
      for (int i = 0; i < table->len; i++) {
        current = &((table->tbl)[i]);
        // printf("looked at name: %s\n", current->name);
        if (strcmp(current->name, name) == 0) {
          return current->addr;
        }
      }
    }
    return -1;   
}

/* Writes the SymbolTable TABLE to OUTPUT. You should use write_symbol() to
   perform the write. Do not print any additional whitespace or characters.
 */
void write_table(SymbolTable* table, FILE* output) {
    if (!table) {
      return;
    }
    Symbol* current = table->tbl;
    if (!current) {
      return;
    } else {
      for (int i = 0; i < table->len; i++) {
        current = &(table->tbl)[i];
        write_symbol(output, current->addr, current->name);
      }
    }
}
