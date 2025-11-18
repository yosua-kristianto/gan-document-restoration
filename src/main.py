from dotenv import load_dotenv

def main():
    """
    This is the main function of this project.
    :return int
    """
    load_dotenv()

    # Setup TensorLayerX to force use TensorFlow
    import os
    os.environ["TL_BACKEND"] = "tensorflow"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


    import tensorlayerx as tlx
    tlx.set_device("GPU")


if __name__ == "__main__":
    from datetime import datetime

    print(f"""
  ________    _____    _______                                                    
 /  _____/   /  _  \   \      \                                                   
/   \  ___  /  /_\  \  /   |   \                                                  
\    \_\  \/    |    \/    |    \                                                 
 \______  /\____|__  /\____|__  /                                                 
        \/         \/         \/                                                  
   ________                                             __                        
   \______ \   ____   ____  __ __  _____   ____   _____/  |_                      
    |    |  \ /  _ \_/ ___\|  |  \/     \_/ __ \ /    \   __\                     
    |    `   (  <_> )  \___|  |  /  Y Y  \  ___/|   |  \  |                       
   /_______  /\____/ \___  >____/|__|_|  /\___  >___|  /__|                       
           \/            \/            \/     \/     \/                           
                                                 
Implementation of Research Titled
"Beyond OCR: GAN-Driven Restoration of Severely Degrading Document"
Session {datetime.now()}
        """)

    main()