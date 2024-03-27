from torch import nn

class ImageEmbeddingModel:
    
    @staticmethod
    def get_model(name, pretrained, embedding_only = True):

        # Resnet models
        if name.startswith('resnet'):
            m, ez = ImageEmbeddingModel._resnet_model(name, pretrained)
            if embedding_only:
                m = nn.Sequential(*list(m.children())[:-1])
            return m, ez
        
        # Efficientnet models
        elif name.startswith('efficientnet'):
            m, ez = ImageEmbeddingModel._efficientnet_model(name, pretrained)
            if embedding_only:
                m = nn.Sequential(*list(m.children())[:-2]) 
            return m, ez
        
        # VGG models
        elif name.startswith('vgg'):
            m, ez = ImageEmbeddingModel._vgg_model(name, pretrained)
            if embedding_only:
                m = nn.Sequential(*list(m.features.children()))
            return m, ez
        
        # Other models
        else:
            raise ValueError(f'The model {name} is not supported.')
    

    @staticmethod
    def _resnet_model(name, pretrained):
        if name == 'resnet18':
            return ImageEmbeddingModel._resnet18(pretrained)
        elif name == 'resnet34':
            return ImageEmbeddingModel._resnet34(pretrained)
        elif name == 'resnet50':
            return ImageEmbeddingModel._resnet50(pretrained)
        elif name == 'resnet101':
            return ImageEmbeddingModel._resnet101(pretrained)
        elif name == 'resnet152':
            return ImageEmbeddingModel._resnet152(pretrained)
        else:
            raise ValueError(f'The model {name} is not supported. (Only: 18, 34, 50, 101 or 152)')
    

    @staticmethod
    def _efficientnet_model(name, pretrained):
        if name == 'efficientnetb0':
            return ImageEmbeddingModel._efficientnetb0(pretrained)
        elif name == 'efficientnetb1':
            return ImageEmbeddingModel._efficientnetb1(pretrained)
        elif name == 'efficientnetb2':
            return ImageEmbeddingModel._efficientnetb2(pretrained)
        elif name == 'efficientnetb3':
            return ImageEmbeddingModel._efficientnetb3(pretrained)
        elif name == 'efficientnetb4':
            return ImageEmbeddingModel._efficientnetb4(pretrained)
        elif name == 'efficientnetb5':
            return ImageEmbeddingModel._efficientnetb5(pretrained)
        elif name == 'efficientnetb6':
            return ImageEmbeddingModel._efficientnetb6(pretrained)
        elif name == 'efficientnetb7':
            return ImageEmbeddingModel._efficientnetb7(pretrained)
        else:
            raise ValueError(f'The model {name} is not supported. (Only: b0, b1, b2, b3, b4, b5, b6 or b7)')


    @staticmethod
    def _vgg_model(name, pretrained):
        if name == 'vgg11':
            return ImageEmbeddingModel._vgg11(pretrained)
        elif name == 'vgg13':
            return ImageEmbeddingModel._vgg13(pretrained)
        elif name == 'vgg16':
            return ImageEmbeddingModel._vgg16(pretrained)
        elif name == 'vgg19':
            return ImageEmbeddingModel._vgg19(pretrained)
        else:
            raise ValueError(f'The model {name} is not supported. (Only: 11, 13, 16 or 19)')
        
        
    @staticmethod
    def _resnet18(pretrained: bool) -> tuple:
        from torchvision.models import resnet18
        model = resnet18()
        if pretrained:
            from torchvision.models import ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            embedding_size = 512
        return model, embedding_size
    

    @staticmethod
    def _resnet34(pretrained: bool) -> tuple:
        from torchvision.models import resnet34
        model = resnet34()
        if pretrained:
            from torchvision.models import ResNet34_Weights
            model = resnet34(weights=ResNet34_Weights.DEFAULT)
            embedding_size = 512
        return model, embedding_size
    

    @staticmethod
    def _resnet50(pretrained: bool) -> tuple:
        from torchvision.models import resnet50
        model = resnet50()
        if pretrained:
            from torchvision.models import ResNet50_Weights
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            embedding_size = 2048
        return model, embedding_size
    

    @staticmethod
    def _resnet101(pretrained: bool) -> tuple:
        from torchvision.models import resnet101
        model = resnet101()
        if pretrained:
            from torchvision.models import ResNet101_Weights
            model = resnet101(weights=ResNet101_Weights.DEFAULT)
            embedding_size = 2048
        return model, embedding_size
    

    @staticmethod
    def _resnet152(pretrained: bool) -> tuple:
        from torchvision.models import resnet152
        model = resnet152()
        if pretrained:
            from torchvision.models import ResNet152_Weights
            model = resnet152(weights=ResNet152_Weights.DEFAULT)
            embedding_size = 2048
        return model, embedding_size
    

    @staticmethod
    def _efficientnetb0(pretrained: bool) -> tuple:
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0()
        if pretrained:
            from torchvision.models import EfficientNetB0_Weights
            model = efficientnet_b0(weights=EfficientNetB0_Weights.DEFAULT)
            embedding_size = 1280
        return model, embedding_size
    

    @staticmethod
    def _efficientnetb1(pretrained: bool) -> tuple:
        from torchvision.models import efficientnet_b1
        model = efficientnet_b1()
        if pretrained:
            from torchvision.models import EfficientNetB1_Weights
            model = efficientnet_b1(weights=EfficientNetB1_Weights.DEFAULT)
            embedding_size = 1280
        return model, embedding_size
    

    @staticmethod
    def _efficientnetb2(pretrained: bool) -> tuple:
        from torchvision.models import efficientnet_b2
        model = efficientnet_b2()
        if pretrained:
            from torchvision.models import EfficientNetB2_Weights
            model = efficientnet_b2(weights=EfficientNetB2_Weights.DEFAULT)
            embedding_size = 1408
        return model, embedding_size
    
    
    @staticmethod
    def _efficientnetb3(pretrained: bool) -> tuple:
        from torchvision.models import efficientnet_b3
        model = efficientnet_b3()
        if pretrained:
            from torchvision.models import EfficientNetB3_Weights
            model = efficientnet_b3(weights=EfficientNetB3_Weights.DEFAULT)
            embedding_size = 1536
        return model, embedding_size
    

    @staticmethod
    def _efficientnetb4(pretrained: bool) -> tuple:
        from torchvision.models import efficientnet_b4
        model = efficientnet_b4()
        if pretrained:
            from torchvision.models import EfficientNetB4_Weights
            model = efficientnet_b4(weights=EfficientNetB4_Weights.DEFAULT)
            embedding_size = 1792
        return model, embedding_size


    @staticmethod
    def _efficientnetb5(pretrained: bool) -> tuple:
        from torchvision.models import efficientnet_b5
        model = efficientnet_b5()
        if pretrained:
            from torchvision.models import EfficientNetB5_Weights
            model = efficientnet_b5(weights=EfficientNetB5_Weights.DEFAULT)
            embedding_size = 2048
        return model, embedding_size
    

    @staticmethod
    def _efficientnetb6(pretrained: bool) -> tuple:
        from torchvision.models import efficientnet_b6
        model = efficientnet_b6()
        if pretrained:
            from torchvision.models import EfficientNetB6_Weights
            model = efficientnet_b6(weights=EfficientNetB6_Weights.DEFAULT)
            embedding_size = 2304
        return model, embedding_size


    @staticmethod
    def _efficientnetb7(pretrained: bool) -> tuple:
        from torchvision.models import efficientnet_b7
        model = efficientnet_b7()
        if pretrained:
            from torchvision.models import EfficientNetB7_Weights
            model = efficientnet_b7(weights=EfficientNetB7_Weights.DEFAULT)
            embedding_size = 2560
        return model, embedding_size
    

    @staticmethod
    def _vgg11(pretrained: bool) -> tuple:
        from torchvision.models import vgg11
        model = vgg11()
        if pretrained:
            from torchvision.models import VGG11_Weights
            model = vgg11(weights=VGG11_Weights.DEFAULT)
            embedding_size = 512
        return model, embedding_size
    

    @staticmethod
    def _vgg13(pretrained: bool) -> tuple:
        from torchvision.models import vgg13
        model = vgg13()
        if pretrained:
            from torchvision.models import VGG13_Weights
            model = vgg13(weights=VGG13_Weights.DEFAULT)
            embedding_size = 512
        return model, embedding_size


    @staticmethod
    def _vgg16(pretrained: bool) -> tuple:
        from torchvision.models import vgg16
        model = vgg16()
        if pretrained:
            from torchvision.models import VGG16_Weights
            model = vgg16(weights=VGG16_Weights.DEFAULT)
            embedding_size = 512
        return model, embedding_size


    @staticmethod
    def _vgg19(pretrained: bool) -> tuple:
        from torchvision.models import vgg19
        model = vgg19()
        if pretrained:
            from torchvision.models import VGG19_Weights
            model = vgg19(weights=VGG19_Weights.DEFAULT)
            embedding_size = 512
        return model, embedding_size