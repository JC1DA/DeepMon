/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "dm.hpp"

namespace deepmon {
    static DeepMon *dm;

    DeepMon &DeepMon::Get() {
        if(dm == NULL) {
            dm = new DeepMon();
        }

        return *dm;
    }

    DeepMon &DeepMon::Get(std::string package_path) {
        if(dm == NULL) {
            dm = new DeepMon(package_path);
        }

        return *dm;
    }
}



