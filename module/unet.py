import tensorflow as tf

def make_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.005)):

	num_classes = 1

	inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

	#Contraction path
	c1 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
	c1 = tf.keras.layers.Dropout(0.1)(c1)
	c1 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
	b1 = tf.keras.layers.BatchNormalization()(c1)
	r1 = tf.keras.layers.ReLU()(b1)
	p1 = tf.keras.layers.MaxPooling2D((2, 2))(r1)

	c2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
	c2 = tf.keras.layers.Dropout(0.1)(c2)
	c2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
	b2 = tf.keras.layers.BatchNormalization()(c2)
	r2 = tf.keras.layers.ReLU()(b2)
	p2 = tf.keras.layers.MaxPooling2D((2, 2))(r2)
	 
	c3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
	c3 = tf.keras.layers.Dropout(0.2)(c3)
	c3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
	b3 = tf.keras.layers.BatchNormalization()(c3)
	r3 = tf.keras.layers.ReLU()(b3)
	p3 = tf.keras.layers.MaxPooling2D((2, 2))(r3)
	 
	c4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
	c4 = tf.keras.layers.Dropout(0.2)(c4)
	c4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
	b4 = tf.keras.layers.BatchNormalization()(c4)
	r4 = tf.keras.layers.ReLU()(b4)
	p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(r4)
	 
	c5 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
	b5 = tf.keras.layers.BatchNormalization()(c5)
	r5 = tf.keras.layers.ReLU()(b5)
	c5 = tf.keras.layers.Dropout(0.3)(r5)
	c5 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

	#Expansive path 
	u6 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
	u6 = tf.keras.layers.concatenate([u6, c4])
	u6 = tf.keras.layers.BatchNormalization()(u6)
	u6 = tf.keras.layers.ReLU()(u6)

	 
	u7 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u6)
	u7 = tf.keras.layers.concatenate([u7, c3])
	u7 = tf.keras.layers.BatchNormalization()(u7)
	u7 = tf.keras.layers.ReLU()(u7)

	 
	u8 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(u7)
	u8 = tf.keras.layers.concatenate([u8, c2])
	u8 = tf.keras.layers.BatchNormalization()(u8)
	u8 = tf.keras.layers.ReLU()(u8)
	 
	u9 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(u8)
	u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
	u9 = tf.keras.layers.BatchNormalization()(u9)
	u9 = tf.keras.layers.ReLU()(u9)

	 
	outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(u9)


	model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
	model.compile(optimizer=optimizer, loss=loss, metrics=[loss, 'accuracy'])
	return model


