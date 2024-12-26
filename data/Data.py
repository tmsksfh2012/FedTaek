import syft as sy

class DataSetEntity:
    def __init__(self, name, description, asset_list, summary=None, citation=None, url=None):
        self.name = name
        self.description = description
        self.summary = summary
        self.citation = citation
        self.url = url
        self.asset_list = asset_list
        
    # def get_name(self):
    #     return self.name
    # def get_description(self):
    #     return self.description
    # def get_summary(self):
    #     return self.summary
    # def get_citation(self):
    #     return self.citation
    # def get_url(self):
    #     return self.url
    # def get_asset_list(self):
    #     return self.asset_list
    
        
class DataAssetEntity:
    def __init__(self, name, data, mock):
        self.name = name
        self.data = data
        self.mock = mock
        
    # def get_name(self):
    #     return self.name
    # def get_data(self):
    #     return self.data
    # def get_mock(self):
    #     return self.mock

class DataHelper:
    def __init__(self, user):
        # """
        # user: PySyft client
        # """
        self.user = user
        
    def upload_dataset(self, dataset_entity: DataSetEntity):
        asset_list = []
        for asset_e in dataset_entity.asset_list:
            asset_list.append(
                sy.Asset(
                    name=asset_e.name,
                    description=f"Asset {asset_e.name}",
                    data=asset_e.data,
                    mock=asset_e.mock
                )
            )

        dataset_kwargs = {
            "name": dataset_entity.name,
            "description": dataset_entity.description,
            "asset_list": asset_list,
        }
        if dataset_entity.summary:
            dataset_kwargs["summary"] = dataset_entity.summary
        if dataset_entity.url:
            dataset_kwargs["url"] = dataset_entity.url
        if dataset_entity.citation:
            dataset_kwargs["citation"] = dataset_entity.citation

        dataset_obj = sy.Dataset(**dataset_kwargs)
        self.user.upload_dataset(dataset_obj)


    def get_dataset_by_name(self, dataset_name):
        return self.user.datasets[dataset_name]
        
    def add_asset(self, dataset: DataSetEntity, asset: DataSetEntity):
        dataset.add_asset(asset)

    def delete_asset(self, dataset: DataSetEntity, asset: DataSetEntity):
        self.user.api.dataset.delete(uid = self.user.datasets[0].id)

    def cast_asset_type(self, asset: DataAssetEntity):
        return sy.Asset(name=asset.get_name(), data=asset.get_data(), mock=asset.get_mock())

    def cast_dataset_type(self, dataset: DataSetEntity, asset_list):
        kwargs = {
            "name": dataset.get_name(),
            "description": dataset.get_description(),
            "asset_list": asset_list,
        }
        if dataset.get_summary() is not None:
            kwargs["summary"] = dataset.get_summary()
        if dataset.get_url() is not None:
            kwargs["url"] = dataset.get_url()
        if dataset.get_citation() is not None:
            kwargs["citation"] = dataset.get_citation()
    
        return sy.Dataset(**kwargs)

    def delete_dataset(self, idx):
        self.user.api.dataset.delete(uid = self.user.datasets[idx].id)

    def delete_code(self, idx):
        self.user.api.code.delete(uid = self.user.code[idx].id)