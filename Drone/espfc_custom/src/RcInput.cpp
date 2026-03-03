#include "RcInput.h"
#include "Config.h"
#include "Debug.h"

int channels[CHANNELS];
bool valid = false;

void RcInput::process(const char* payload) {
  valid = false;
  char* token;
  char* str = strdup(payload);
  int i = 0;
  token = strtok(str, ",");
  while (token && i < CHANNELS) {
    channels[i++] = atoi(token);
    token = strtok(NULL, ",");
  }
  free(str);
  valid = true;
  Debug::log("[RC] Throttle="); Debug::logln(channels[THROTTLE_CH]);
}

bool RcInput::isValid() { return valid; }
int RcInput::get(int ch) { return channels[ch]; }
