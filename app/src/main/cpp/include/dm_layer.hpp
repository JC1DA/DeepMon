#ifndef DM_LAYER_HPP
#define DM_LAYER_HPP

#include <string>
#include "dm_common.hpp"
#include "dm_blob.hpp"
#include <queue>

using namespace std;

namespace deepmon {
	class DM_Layer {
	private:
        //caching data
        int total_non_cached_blocks = 0;
        int non_cached_blocks_indices_x[64];
        int non_cached_blocks_indices_y[64];
        int side_x = 8;
        int side_y = 8;
        DM_Blob *cached_result = NULL;
	public:
		DM_Layer(string name, string type, vector<string> bottom_layers, MEMORY_LAYOUT mem_layout) {
            this->name = name;
            this->type = type;
            this->bottom_layers = bottom_layers;
            this->mem_layout = mem_layout;
        }

        bool IsCorrupted() {
            return corrupted;
        }
        bool IsUsingPersistentBlob() {
            return persistant_blobs;
        }
        string GetName() {
            return this->name;
        }
        string GetType() {
            return this->type;
        }
        vector<string> GetBottomLayersNames() {
            return vector<string>(bottom_layers);
        }
        vector<string> GetTopLayersNames() {
            return vector<string>(top_layers);
        }
        vector<uint32_t> GetOutputShapes() {
            return vector<uint32_t>(output_shapes);
        }
		virtual void LoadWeights() = 0;
        virtual void PrintInfo() = 0;
        virtual void ComputeOutputShapes(vector<vector<uint32_t >> inputs_shapes_no_batches) = 0;

        void AppendTopLayer(string layer_name) {
            top_layers.push_back(layer_name);
        }
        void EnqueueInputBlob(DM_Blob *input) {
            this->input_queue.push(input);
        }
        DM_Blob *Forward() {
            DM_Blob *result = NULL;

            vector<DM_Blob *> input_blobs;
            for(int i = 0 ; i < input_queue.size() ; i++) {
                DM_Blob *input = input_queue.front();
                input_queue.pop();

                if(this->env != input->get_env()) {
                    //convert to correct environment
                    DM_Blob *converted_input = NULL;
                    if(this->env == ENVIRONMENT_CPU)
                        converted_input = input->ConvertToCpuBlob();
                    else if(this->env == ENVIRONMENT_GPU)
                        converted_input = input->CovnertToGpuBlob(this->precision);

                    if(!input->is_persistent_blob()) {
                        delete input;
                    }

                    input = converted_input;
                }

                input_blobs.push_back(input);
            }

            if(env == ENVIRONMENT_CPU) {
                //call forward_cpu
                result = ForwardCpu(input_blobs);
            } else if(env == ENVIRONMENT_GPU) {
                //call forward_gpu
                result = ForwardGpu(input_blobs);
            } else {
                LOGE("Incorrect Environment");
                return NULL;
            }

            //delete inputs
            for(int i = 0 ; i < input_blobs.size() ; i++) {
                DM_Blob *input = input_blobs[i];
                if(!input->is_persistent_blob())
                    delete input;
            }
            input_blobs.clear();

            if(this->persistant_blobs)
                result->set_persistent(true);

            return result;
        }
        DM_Blob *ForwardCache() {
            DM_Blob *result = NULL;

            vector<DM_Blob *> input_blobs;
            for(int i = 0 ; i < input_queue.size() ; i++) {
                DM_Blob *input = input_queue.front();
                input_queue.pop();

                if(this->env != input->get_env()) {
                    //convert to correct environment
                    DM_Blob *converted_input = NULL;
                    if(this->env == ENVIRONMENT_CPU)
                        converted_input = input->ConvertToCpuBlob();
                    else if(this->env == ENVIRONMENT_GPU)
                        converted_input = input->CovnertToGpuBlob(this->precision);

                    if(!input->is_persistent_blob()) {
                        delete input;
                    }

                    input = converted_input;
                }

                input_blobs.push_back(input);
            }

            if(env == ENVIRONMENT_CPU) {
                //call forward_cpu
                result = ForwardCache(input_blobs);
            } else if(env == ENVIRONMENT_GPU) {
                //call forward_gpu
                result = ForwardCache(input_blobs);
            } else {
                LOGE("Incorrect Environment");
                return NULL;
            }

            //delete inputs
            for(int i = 0 ; i < input_blobs.size() ; i++) {
                DM_Blob *input = input_blobs[i];
                if(!input->is_persistent_blob())
                    delete input;
            }
            input_blobs.clear();

            if(this->persistant_blobs)
                result->set_persistent(true);

            return result;
        }
        bool IsCachable() {
            if(type.compare(LAYER_NAME_CONV))
                return false;
            if(GetOutputShapes()[0] % (side_y) == 0 && GetOutputShapes()[1] % (side_x) == 0)
                return true;
            return false;
        }
        int GetCachingBlockSizeX() {
            return output_shapes[0] / side_y;
        }
        int GetCachingBlockSizeY() {
            return output_shapes[1] / side_x;
        }
        int *GetCachingNonCachedBlockIndicesX() {
            return this->non_cached_blocks_indices_x;
        }
        int *GetCachingNonCachedBlockIndicesY() {
            return this->non_cached_blocks_indices_y;
        }
        int GetCachingTotalNonCachedBlocks() {
            return this->total_non_cached_blocks;
        }
        int GetCachingSideX() {
            return this->side_x;
        }
        int GetCachingSideY() {
            return this->side_y;
        }
        DM_Blob *GetCachingResult() {
            return this->cached_result;
        }
        void DeleteCachingResult() {
            delete this->cached_result;
            this->cached_result = NULL;
        }
        void StoreCachingResult(DM_Blob *result) {
            this->cached_result = result->CovnertToGpuBlob(this->precision);
        }
        void SetUpCaching(int total_non_cached_blocks, int *non_cached_indices_x, int *non_cached_indices_y) {
            for(int i = 0 ; i < total_non_cached_blocks; i++) {
                this->non_cached_blocks_indices_x[i] = non_cached_indices_x[i];
                this->non_cached_blocks_indices_y[i] = non_cached_indices_y[i];
            }
            this->total_non_cached_blocks = total_non_cached_blocks;
        }
	protected:
        string name;
        string type;
        bool persistant_blobs = false;
        bool corrupted = false;
        vector<string> bottom_layers;
        vector<string> top_layers;
		ENVIRONMENT_TYPE env = ENVIRONMENT_CPU;
        PRESICION_TYPE precision = PRECISION_32;
        MEMORY_LAYOUT mem_layout = MEMORY_LAYOUT_DM;
        vector<vector<uint32_t>> inputs_shapes; //only used in some layers
        vector<uint32_t> output_shapes;
        queue<DM_Blob *> input_queue;
        virtual DM_Blob *ForwardCpu(vector<DM_Blob *> blobs) = 0;
        virtual DM_Blob *ForwardGpu(vector<DM_Blob *> blobs) = 0;
        virtual DM_Blob *ForwardCache(vector<DM_Blob *> blobs) = 0;
	};
}

#endif