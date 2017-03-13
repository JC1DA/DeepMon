#ifndef DM_LAYER_HPP
#define DM_LAYER_HPP

#include <string>
#include "dm_blob.hpp"
#include "dm_layer_param.hpp"

namespace deepmon {
	class DM_Layer {
	private:

	public:
		explicit DM_Layer(DM_Layer_Param &param);
		virtual void Forward(
			    const std::vector<DM_Blob *> &bottom,
			    const std::vector<DM_Blob *> &top);
        virtual void LayerSetUp(
                const std::vector<DM_Blob*>& bottom,
                const std::vector<DM_Blob*>& top) {
        };
        virtual void Reshape(
                const std::vector<DM_Blob*>& bottom,
                const std::vector<DM_Blob*>& top
        ) = 0;
	protected:
		virtual void Forward_CPU(
			const std::vector<DM_Blob *> &bottom,
			const std::vector<DM_Blob *> &top
			) = 0;
        virtual void Forward_GPU(
                const std::vector<DM_Blob *> &bottom,
                const std::vector<DM_Blob *> &top
        ) = 0;
	};
}

#endif