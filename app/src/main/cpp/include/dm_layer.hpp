#ifndef DM_LAYER_HPP
#define DM_LAYER_HPP

#include <string>
#include "dm_blob.hpp"
#include "dm_layer_param.hpp"

using namespace std;

namespace deepmon {
	class DM_Layer {
	private:
	public:
		explicit DM_Layer();
		virtual void Forward(
			    const std::vector<DM_Blob *> &bottom,
			    const std::vector<DM_Blob *> &top) {
            switch(env) {
                case ENVIRONMENT_CPU:
                    Forward_CPU(bottom, top);
                    break;
                case ENVIRONMENT_GPU:
                    Forward_GPU(bottom, top);
                    break;
            }
        };
        virtual void LayerSetUp(
                const std::vector<DM_Blob*>& bottom,
                const std::vector<DM_Blob*>& top) {
        };
        virtual void Reshape(
                const std::vector<DM_Blob*>& bottom,
                const std::vector<DM_Blob*>& top
        ) = 0;
        bool IsCorrupted() {
            return corrupted;
        }
	protected:
		virtual void Forward_CPU(
			const std::vector<DM_Blob *> &bottom,
			const std::vector<DM_Blob *> &top
			) = 0;
        virtual void Forward_GPU(
                const std::vector<DM_Blob *> &bottom,
                const std::vector<DM_Blob *> &top
        ) = 0;

		ENVIRONMENT_TYPE env;
        bool corrupted = false;
        vector<DM_Blob *> bottom_blobs; //only store references
        vector<DM_Blob *> top_blobs; //store physical blobs
	};
}

#endif